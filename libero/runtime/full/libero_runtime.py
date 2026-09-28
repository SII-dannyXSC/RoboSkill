#!/usr/bin/env python3
"""Trusted LIBERO allocation, lease proxy, retries, and artifact recording."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import threading
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


TRANSIENT_HTTP_STATUSES = {502, 503, 504}
DEPTH_PNG_ENCODING = "png-u16-linear"
DEPTH_PNG_BASE_SCALE_M_PER_UNIT = 0.0001


def _encode_depth_png_u16(raw: bytes, shape: list[int]) -> tuple[bytes, dict[str, Any]]:
    """Quantize metric float32 depth and encode it as a lossless 16-bit PNG."""

    if len(shape) != 2:
        raise RuntimeError(f"PNG depth storage requires a 2-D shape, got {shape!r}")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - deployment dependency failure
        raise RuntimeError("PNG depth storage requires numpy and Pillow") from exc

    values = np.frombuffer(raw, dtype="<f4").reshape(tuple(shape))
    if not np.all(np.isfinite(values)):
        raise RuntimeError("depth frame contains non-finite values")

    min_depth_m = float(values.min()) if values.size else 0.0
    max_depth_m = float(values.max()) if values.size else 0.0
    if min_depth_m < 0.0:
        raise RuntimeError(f"depth frame contains a negative value: {min_depth_m}")

    scale_m_per_unit = max(
        DEPTH_PNG_BASE_SCALE_M_PER_UNIT,
        max_depth_m / 65535.0 if max_depth_m else DEPTH_PNG_BASE_SCALE_M_PER_UNIT,
    )
    quantized = np.clip(np.rint(values / scale_m_per_unit), 0, 65535).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(quantized).save(buffer, format="PNG", compress_level=6)

    return buffer.getvalue(), {
        "encoding": DEPTH_PNG_ENCODING,
        "dtype": "uint16",
        "source_dtype": "float32",
        "unit": "meter",
        "scale_m_per_unit": scale_m_per_unit,
        "offset_m": 0.0,
        "quantization_max_error_m": scale_m_per_unit / 2.0,
        "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "min_depth_m": min_depth_m,
        "max_depth_m": max_depth_m,
    }


class GatewayError(RuntimeError):
    """An HTTP response from the remote LIBERO Gateway was not successful."""

    def __init__(self, status: int, payload: dict[str, Any]):
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        code = error.get("code", "UNKNOWN") if isinstance(error, dict) else "UNKNOWN"
        super().__init__(f"LIBERO Gateway HTTP {status}: {code}")
        self.status = status
        self.payload = payload


def atomic_bytes(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Write one file atomically and durably with private permissions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str, mode: int = 0o600) -> None:
    """Write one UTF-8 text file atomically and durably."""

    atomic_bytes(path, value.encode("utf-8"), mode=mode)


def atomic_json(path: Path, value: Any, mode: int = 0o600) -> None:
    """Write one JSON file atomically and durably."""

    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        mode=mode,
    )


def append_jsonl(path: Path, value: dict[str, Any], mode: int = 0o600) -> None:
    """Append one durable JSONL commit record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, mode)
    try:
        data = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class GatewayClient:
    """Trusted raw HTTP client that preserves request identity across retries."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float = 30.0,
        retry_initial: float = 0.5,
        retry_max: float = 12.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.retry_initial = retry_initial
        self.retry_max = retry_max
        self.stop_event = stop_event or threading.Event()
        # The deployed Gateway is reached through a local SSH tunnel. Bypass
        # process-wide proxy variables so localhost can never be sent to a proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        token: str | None = None,
        idempotency_key: str | None = None,
        retry_statuses: set[int] | None = None,
        max_retry_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Send one logical request until it succeeds or the controller stops."""

        encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        retryable = TRANSIENT_HTTP_STATUSES if retry_statuses is None else retry_statuses
        delay = self.retry_initial
        deadline = None if max_retry_seconds is None else time.monotonic() + max_retry_seconds
        while not self.stop_event.is_set():
            headers = {"Accept": "application/json"}
            if encoded is not None:
                headers["Content-Type"] = "application/json"
            if token is not None:
                headers["Authorization"] = f"Bearer {token}"
            if idempotency_key is not None:
                headers["Idempotency-Key"] = idempotency_key
            request = urllib.request.Request(
                self.base_url + path,
                data=encoded,
                headers=headers,
                method=method,
            )
            try:
                with self._opener.open(request, timeout=self.request_timeout) as response:
                    text = response.read().decode("utf-8")
                    value = {} if not text else json.loads(text)
                    if not isinstance(value, dict):
                        raise RuntimeError("LIBERO Gateway returned a non-object JSON response")
                    return value
            except urllib.error.HTTPError as error:
                try:
                    value = json.loads(error.read().decode("utf-8"))
                except Exception:
                    value = {"error": {"code": "INVALID_ERROR_RESPONSE"}}
                error_value = value.get("error") if isinstance(value, dict) else None
                code = error_value.get("code") if isinstance(error_value, dict) else None
                if error.code not in retryable and code != "SESSION_BUSY":
                    raise GatewayError(error.code, value) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError):
                pass
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError(f"LIBERO request did not recover within {max_retry_seconds} seconds")
            if self.stop_event.wait(delay):
                break
            delay = min(delay * 2, self.retry_max)
        raise RuntimeError("LIBERO request stopped because the trusted controller is shutting down")

    def allocate(self, experiment: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        """Create one remote Run and Session without exposing its token."""

        return self.request(
            "POST",
            "/v2/experiments",
            body=experiment,
            idempotency_key=idempotency_key,
            retry_statuses={429, 502, 503, 504},
        )

    def wait_ready(self, session_id: str, token: str) -> dict[str, Any]:
        """Wait until an allocated Session reaches ready."""

        while not self.stop_event.is_set():
            status = self.request("GET", f"/v2/sessions/{session_id}", token=token)
            state = status.get("state")
            if state == "ready":
                return status
            if state not in {"created", "starting"}:
                raise RuntimeError(f"LIBERO Session entered unexpected state {state!r}")
            if self.stop_event.wait(0.25):
                break
        raise RuntimeError("LIBERO Session startup stopped")


class ArtifactRecorder:
    """Persist every proxy Reset and Step independently of Agent-written code."""

    def __init__(self, workspace: Path, manifest: dict[str, Any]) -> None:
        self.root = workspace / "libero-artifacts"
        if self.root.exists() and any(self.root.iterdir()):
            raise RuntimeError(f"Workspace already contains LIBERO artifacts: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.root / "episodes").mkdir(mode=0o700)
        self.depth_storage = os.environ.get("LIBERO_DEPTH_STORAGE", "png-u16").strip().lower()
        if self.depth_storage not in {"png-u16", "raw-float32"}:
            raise RuntimeError(
                "LIBERO_DEPTH_STORAGE must be 'png-u16' or 'raw-float32', "
                f"got {self.depth_storage!r}"
            )
        atomic_json(
            self.root / "manifest.json",
            {
                "schema_version": 1,
                "layout_version": "libero-prototype-v2",
                "frame_commit_index": "frames.jsonl",
                "image_bytes": "gateway_original_jpeg",
                "depth_storage": {
                    "mode": self.depth_storage,
                    "encoding": (
                        DEPTH_PNG_ENCODING if self.depth_storage == "png-u16" else "raw-float32"
                    ),
                    "base_scale_m_per_unit": (
                        DEPTH_PNG_BASE_SCALE_M_PER_UNIT
                        if self.depth_storage == "png-u16"
                        else None
                    ),
                    "source_dtype": "float32",
                    "unit": "meter",
                },
                **manifest,
            },
        )
        self._lock = threading.Lock()
        self._committed: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _extract_image_bytes(value: Any) -> bytes | None:
        encoded = value if isinstance(value, str) else value.get("base64") if isinstance(value, dict) else None
        if not isinstance(encoded, str):
            return None
        return base64.b64decode(encoded, validate=True)

    @staticmethod
    def _safe_component(value: Any) -> str:
        """Convert an external camera key into one safe path component."""

        source = str(value)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip(".")
        if not safe or safe in {".", ".."}:
            raise RuntimeError(f"unsafe LIBERO camera key: {source!r}")
        return safe

    @staticmethod
    def _extract_depth(value: Any) -> tuple[bytes, list[int], str, str] | None:
        """Decode one public metric depth payload into its raw float32 bytes."""

        if not isinstance(value, dict) or value.get("encoding") != "float32-zlib-base64":
            return None
        encoded = value.get("base64")
        shape = value.get("shape")
        dtype = value.get("dtype")
        unit = value.get("unit")
        if (
            not isinstance(encoded, str)
            or not isinstance(shape, list)
            or not shape
            or any(type(size) is not int or size < 1 for size in shape)
            or dtype != "float32"
            or unit != "meter"
        ):
            raise RuntimeError("invalid LIBERO metric depth payload")
        try:
            raw = zlib.decompress(base64.b64decode(encoded, validate=True))
        except (ValueError, zlib.error) as error:
            raise RuntimeError("invalid LIBERO metric depth encoding") from error
        expected = math.prod(shape) * 4
        if len(raw) != expected:
            raise RuntimeError(
                f"LIBERO metric depth byte length {len(raw)} does not match shape {shape}"
            )
        return raw, shape, dtype, unit

    def record(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        action: list[float] | None = None,
        expected_step_index: int | None = None,
        idempotency_key: str,
        episode_index: int,
    ) -> dict[str, Any]:
        """Commit one Frame before returning paths and persisted=true to the Agent."""

        with self._lock:
            result = json.loads(json.dumps(payload))
            observation = result.get("observation")
            if not isinstance(observation, dict):
                raise RuntimeError("LIBERO response is missing observation")
            if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 1:
                raise RuntimeError("LIBERO Frame requires a positive trusted episode_index")
            payload_episode_index = result.get("episode_index")
            if payload_episode_index is not None and payload_episode_index != episode_index:
                raise RuntimeError("LIBERO response episode_index conflicts with trusted lease state")
            step_index = int(observation.get("step_index", 0))
            episode = self.root / "episodes" / f"episode-{episode_index:04d}"
            stem = f"frame-{step_index:05d}_{operation}"
            frame_ref = f"episode-{episode_index:04d}/{stem}"
            response_images: dict[str, dict[str, Any]] = {}
            indexed_images: dict[str, dict[str, Any]] = {}
            used_camera_paths: set[str] = set()
            images = observation.get("images", {})
            if isinstance(images, dict):
                for camera, encoded in sorted(images.items()):
                    data = self._extract_image_bytes(encoded)
                    if data is None:
                        continue
                    camera_key = str(camera)
                    camera_path = self._safe_component(camera_key)
                    if camera_path in used_camera_paths:
                        raise RuntimeError(f"camera path collision after sanitizing {camera_key!r}")
                    used_camera_paths.add(camera_path)
                    image_path = episode / "images" / camera_path / f"{stem}.jpg"
                    relative = str(image_path.relative_to(self.root))
                    digest = hashlib.sha256(data).hexdigest()
                    response_images[camera_key] = {
                        "path": str(image_path),
                        "bytes": len(data),
                        "sha256": digest,
                    }
                    indexed_images[camera_key] = {
                        "path": relative,
                        "bytes": len(data),
                        "sha256": digest,
                    }
            observation["images"] = response_images
            stored_observation = json.loads(json.dumps(observation))
            stored_observation["images"] = indexed_images
            response_depth: dict[str, dict[str, Any]] = {}
            indexed_depth: dict[str, dict[str, Any]] = {}
            encoded_depth_files: dict[str, bytes] = {}
            used_depth_paths: set[str] = set()
            depth = observation.get("depth")
            if isinstance(depth, dict):
                for camera, encoded in sorted(depth.items()):
                    decoded = self._extract_depth(encoded)
                    if decoded is None:
                        continue
                    data, shape, dtype, unit = decoded
                    camera_key = str(camera)
                    camera_path = self._safe_component(camera_key)
                    if camera_path in used_depth_paths:
                        raise RuntimeError(
                            f"depth camera path collision after sanitizing {camera_key!r}"
                        )
                    used_depth_paths.add(camera_path)
                    if self.depth_storage == "png-u16":
                        stored, storage_metadata = _encode_depth_png_u16(data, shape)
                        depth_path = episode / "depth" / camera_path / f"{stem}.depth.png"
                        descriptor = {
                            "shape": shape,
                            **storage_metadata,
                            "bytes": len(stored),
                            "sha256": hashlib.sha256(stored).hexdigest(),
                        }
                    else:
                        stored = data
                        depth_path = episode / "depth" / camera_path / f"{stem}.float32.bin"
                        descriptor = {
                            "shape": shape,
                            "encoding": "raw-float32",
                            "dtype": dtype,
                            "unit": unit,
                            "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                    relative = str(depth_path.relative_to(self.root))
                    encoded_depth_files[camera_key] = stored
                    response_depth[camera_key] = {"path": str(depth_path), **descriptor}
                    indexed_depth[camera_key] = {"path": relative, **descriptor}
                observation["depth"] = response_depth
                stored_observation["depth"] = indexed_depth
            observation_path = episode / "observations" / f"{stem}.json"
            action_path: Path | None = None
            if action is not None:
                action_path = episode / "actions" / f"{stem}.json"
                action_value = {
                    "action": action,
                    "expected_step_index": expected_step_index,
                    "idempotency_key": idempotency_key,
                }
            frame_path = episode / "frame-metadata" / f"{stem}.json"
            frame = {
                "committed_at": time.time(),
                "idempotency_key": idempotency_key,
                "frame_ref": frame_ref,
                "operation": operation,
                "episode_index": episode_index,
                "step_index": step_index,
                "frame_id": observation.get("frame_id", step_index),
                "terminated": result.get("terminated") is True,
                "truncated": result.get("truncated") is True,
                "reward": result.get("reward", 0),
                "observation": str(observation_path.relative_to(self.root)),
                "action": None if action_path is None else str(action_path.relative_to(self.root)),
                "images": indexed_images,
                "depth": indexed_depth,
            }
            if frame_path.exists():
                existing_frame = json.loads(frame_path.read_text(encoding="utf-8"))
                if (
                    existing_frame.get("idempotency_key") != idempotency_key
                    or existing_frame.get("frame_ref") != frame_ref
                    or existing_frame.get("operation") != operation
                ):
                    raise RuntimeError(f"refusing to overwrite a different committed Frame: {frame_path}")
                frame = existing_frame
            index_record = {**frame, "frame_metadata": str(frame_path.relative_to(self.root))}

            def json_bytes(value: Any) -> bytes:
                return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

            planned: dict[Path, bytes] = {}
            for camera, descriptor in indexed_images.items():
                final_path = self.root / descriptor["path"]
                source = images[camera]
                image_data = self._extract_image_bytes(source)
                if image_data is None:
                    raise RuntimeError(f"LIBERO image disappeared while recording {camera!r}")
                planned[final_path] = image_data
            if isinstance(depth, dict):
                for camera, descriptor in indexed_depth.items():
                    final_path = self.root / descriptor["path"]
                    planned[final_path] = encoded_depth_files[camera]
            planned[observation_path] = json_bytes(stored_observation)
            if action_path is not None:
                planned[action_path] = json_bytes(action_value)
            planned[frame_path] = json_bytes(frame)

            # Stage the whole logical Frame first. Moving several files cannot be
            # one filesystem transaction, so retries accept only byte-identical
            # files left by an interrupted commit. frames.jsonl remains the final
            # authoritative commit point.
            partial = self.root / ".partial" / hashlib.sha256(
                idempotency_key.encode("utf-8")
            ).hexdigest()
            partial.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                for final_path, data in planned.items():
                    staged = partial / final_path.relative_to(self.root)
                    atomic_bytes(staged, data)
                for final_path, data in planned.items():
                    if final_path.exists():
                        if final_path.read_bytes() != data:
                            raise RuntimeError(
                                f"existing Frame file differs from idempotent retry: {final_path}"
                            )
                        continue
                    staged = partial / final_path.relative_to(self.root)
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staged, final_path)
                committed = self._committed.get(idempotency_key)
                if committed is None:
                    append_jsonl(self.root / "frames.jsonl", index_record)
                    self._committed[idempotency_key] = index_record
                elif (
                    committed.get("frame_ref") != frame_ref
                    or committed.get("operation") != operation
                ):
                    raise RuntimeError("Idempotency-Key conflicts with a committed Frame")
            finally:
                shutil.rmtree(partial, ignore_errors=True)
            result["local_artifacts"] = {
                "persisted": True,
                "frame_ref": frame_ref,
                "observation": str(observation_path),
                "action": None if action_path is None else str(action_path),
                "frame_metadata": str(frame_path),
                "frame_index": str(self.root / "frames.jsonl"),
                "images": response_images,
                "depth": response_depth,
            }
            return result

    def record_result(self, payload: dict[str, Any]) -> None:
        """Persist the most recent server-authoritative result."""

        atomic_json(self.root / "result.json", {**payload, "queried_at": time.time()})

    def record_closed(self, payload: dict[str, Any]) -> None:
        """Persist trusted cleanup completion without credentials."""

        atomic_json(self.root / "closed.json", {**payload, "recorded_at": time.time()})


@dataclass
class CachedOperation:
    fingerprint: str
    response: dict[str, Any]
    progress: dict[str, Any] | None = None
    progress_reported: bool = False


@dataclass
class Lease:
    """One trusted mapping from a local capability URL to one remote Session."""

    gateway: GatewayClient
    run_id: str
    session_id: str
    token: str
    recorder: ArtifactRecorder
    instruction: str
    step_index: int | None = None
    episode_index: int | None = None
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    operations: dict[str, CachedOperation] = field(default_factory=dict)

    @staticmethod
    def _fingerprint(operation: str, body: dict[str, Any]) -> str:
        encoded = json.dumps([operation, body], sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _validate_action(value: Any) -> list[float]:
        if not isinstance(value, list) or len(value) != 7:
            raise ValueError("action must contain exactly seven values")
        try:
            action = [float(item) for item in value]
        except (TypeError, ValueError) as error:
            raise ValueError("action values must be finite numbers") from error
        if any(not math.isfinite(item) or item < -1 or item > 1 for item in action):
            raise ValueError("action values must be in [-1,1]")
        return action

    def mutate(self, operation: str, body: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        """Execute or recover one idempotent Reset or Step."""

        fingerprint = self._fingerprint(operation, body)
        with self.lock:
            cached = self.operations.get(idempotency_key)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    raise ValueError("Idempotency-Key was reused with a different request")
                if (
                    cached.progress is not None
                    and not cached.progress_reported
                    and self.progress_callback is not None
                ):
                    self.progress_callback(cached.progress)
                    cached.progress_reported = True
                return cached.response
            previous_step_index = self.step_index
            if operation == "reset":
                payload = self.gateway.request(
                    "POST",
                    f"/v2/sessions/{self.session_id}/reset",
                    body={},
                    token=self.token,
                    idempotency_key=idempotency_key,
                )
                remote_episode = payload.get("episode_index")
                if (
                    isinstance(remote_episode, bool)
                    or not isinstance(remote_episode, int)
                    or remote_episode < 1
                ):
                    raise RuntimeError("LIBERO Reset lacks a positive episode_index")
                response = self.recorder.record(
                    "reset",
                    payload,
                    idempotency_key=idempotency_key,
                    episode_index=remote_episode,
                )
            elif operation == "step":
                action = self._validate_action(body.get("action"))
                if self.step_index is None or self.episode_index is None:
                    raise ValueError("reset must succeed before step")
                if "expected_step_index" in body:
                    raise ValueError("expected_step_index is managed by the trusted lease proxy")
                remote_body = {"action": action, "expected_step_index": self.step_index}
                payload = self.gateway.request(
                    "POST",
                    f"/v2/sessions/{self.session_id}/step",
                    body=remote_body,
                    token=self.token,
                    idempotency_key=idempotency_key,
                )
                response = self.recorder.record(
                    "step",
                    payload,
                    action=action,
                    expected_step_index=self.step_index,
                    idempotency_key=idempotency_key,
                    episode_index=self.episode_index,
                )
            else:
                raise ValueError(f"unsupported mutation {operation}")
            observation = response.get("observation", {})
            if not isinstance(observation, dict) or not isinstance(observation.get("step_index"), int):
                raise RuntimeError("LIBERO response lacks an integer observation.step_index")
            self.step_index = observation["step_index"]
            if operation == "reset":
                self.episode_index = remote_episode
            artifacts = response.get("local_artifacts", {})
            frame_ref = artifacts.get("frame_ref") if isinstance(artifacts, dict) else None
            if not isinstance(frame_ref, str) or not frame_ref:
                raise RuntimeError("recorded LIBERO response lacks local_artifacts.frame_ref")
            if self.episode_index is None:
                raise RuntimeError("committed LIBERO response lacks an episode index")
            progress: dict[str, Any] = {
                "progressId": (
                    f"episode-{self.episode_index}:reset"
                    if operation == "reset"
                    else f"episode-{self.episode_index}:step-{self.step_index}"
                ),
                "kind": operation,
                "episodeIndex": self.episode_index,
                "stepIndex": self.step_index,
                "frameRef": frame_ref,
                "committedAt": int(time.time() * 1000),
            }
            if operation == "step":
                if previous_step_index is None:
                    raise RuntimeError("committed Step lacks a previous step index")
                progress["previousStepIndex"] = previous_step_index
            cached = CachedOperation(fingerprint, response, progress=progress)
            self.operations[idempotency_key] = cached
            if self.progress_callback is not None:
                self.progress_callback(progress)
                cached.progress_reported = True
            return response

    def result(self) -> dict[str, Any]:
        """Return and persist the remote server-authoritative result."""

        with self.lock:
            payload = self.gateway.request(
                "GET", f"/v2/sessions/{self.session_id}/result", token=self.token
            )
            self.recorder.record_result(payload)
            return payload


class LeaseProxy:
    """Loopback HTTP proxy exposing one capability path and no remote credential."""

    def __init__(
        self,
        lease: Lease,
        *,
        active: bool = True,
        activation_check: Callable[[], bool] | None = None,
    ) -> None:
        self.lease = lease
        self.secret = secrets.token_urlsafe(24)
        self._active = active
        self._activation_check = activation_check
        self._mutation_check: Callable[[str], tuple[bool, str, str]] | None = None
        self._gate_lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "LiberoLeaseProxy/1"

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded)

            def _operation(self) -> str | None:
                prefix = f"/leases/{proxy.secret}/"
                parsed = urlparse(self.path)
                if not parsed.path.startswith(prefix):
                    return None
                operation = parsed.path[len(prefix):]
                return operation if "/" not in operation else None

            def _body(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as error:
                    raise ValueError("invalid Content-Length") from error
                if length > 1024 * 1024:
                    raise ValueError("request body is too large")
                raw = self.rfile.read(length)
                value = {} if not raw else json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("request body must be one JSON object")
                return value

            def do_POST(self) -> None:  # noqa: N802
                operation = self._operation()
                if operation not in {"reset", "step"}:
                    self._send(404, {"error": {"code": "NOT_FOUND"}})
                    return
                key = self.headers.get("Idempotency-Key", "")
                if not key or len(key) > 200:
                    self._send(400, {"error": {"code": "IDEMPOTENCY_KEY_REQUIRED"}})
                    return
                if not proxy.ensure_active():
                    code = "PREPARE_RESET_LOCKED" if operation == "reset" else "PREPARE_STEP_FORBIDDEN"
                    self._send(
                        409,
                        {"error": {"code": code, "message": "task execution has not passed the atomic PLAN gate"}},
                    )
                    return
                try:
                    admission = proxy.admit_mutation(operation)
                    if admission is not None:
                        code, message = admission
                        self._send(409, {"error": {"code": code, "message": message}})
                        return
                    response = proxy.lease.mutate(operation, self._body(), key)
                    self._send(200, response)
                except ValueError as error:
                    self._send(400, {"error": {"code": "INVALID_REQUEST", "message": str(error)}})
                except GatewayError as error:
                    self._send(error.status, error.payload)
                except Exception as error:
                    self._send(502, {"error": {"code": "PROXY_FAILURE", "message": str(error)}})

            def do_GET(self) -> None:  # noqa: N802
                operation = self._operation()
                if operation == "healthz":
                    self._send(200, {"status": "ok"})
                    return
                if operation != "result":
                    self._send(404, {"error": {"code": "NOT_FOUND"}})
                    return
                try:
                    self._send(200, proxy.lease.result())
                except GatewayError as error:
                    self._send(error.status, error.payload)
                except Exception as error:
                    self._send(502, {"error": {"code": "PROXY_FAILURE", "message": str(error)}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="libero-lease-proxy", daemon=True)

    @property
    def endpoint(self) -> str:
        """Return the Agent-visible capability URL."""

        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/leases/{self.secret}"

    def start(self) -> None:
        """Start serving Agent HTTP requests."""

        self.thread.start()

    def set_activation_check(self, check: Callable[[], bool]) -> None:
        """Install a read-only workflow check used to close the polling race."""

        with self._gate_lock:
            self._activation_check = check

    def set_mutation_check(
        self,
        check: Callable[[str], tuple[bool, str, str]],
    ) -> None:
        """Install an operation-specific admission check after task activation."""

        with self._gate_lock:
            self._mutation_check = check

    def admit_mutation(self, operation: str) -> tuple[str, str] | None:
        """Return one controller-visible denial after task activation, if any."""

        with self._gate_lock:
            check = self._mutation_check
        if check is None:
            return None
        allowed, code, message = check(operation)
        return None if allowed else (code, message)

    def activate(self) -> None:
        """Open Reset/Step after the workflow leaves PLAN."""

        with self._gate_lock:
            self._active = True

    def ensure_active(self) -> bool:
        """Return whether mutation is open, consulting workflow state once if needed."""

        with self._gate_lock:
            if self._active:
                return True
            check = self._activation_check
        if check is not None and check():
            self.activate()
            return True
        return False

    def close(self) -> None:
        """Stop accepting local Agent requests."""

        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def create_lease(
    gateway: GatewayClient,
    workspace: Path,
    experiment: dict[str, Any],
    *,
    allocation_key: str | None = None,
) -> tuple[Lease, dict[str, Any]]:
    """Allocate one remote experiment and construct its trusted local lease."""

    allocation = gateway.allocate(
        experiment,
        allocation_key or f"allocation-{secrets.token_urlsafe(20)}",
    )
    run = allocation.get("run")
    session = allocation.get("session")
    token = allocation.get("agent_token")
    if not isinstance(run, dict) or not isinstance(session, dict) or not isinstance(token, str):
        raise RuntimeError("LIBERO allocation response is incomplete")
    run_id = run.get("run_id")
    session_id = session.get("session_id")
    if not isinstance(run_id, str) or not isinstance(session_id, str):
        raise RuntimeError("LIBERO allocation response lacks Run or Session ID")
    ready = gateway.wait_ready(session_id, token)
    instruction = ready.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError("ready LIBERO Session lacks a task instruction")
    instruction = instruction.strip()
    recorder = ArtifactRecorder(
        workspace,
        {
            "schema_version": 1,
            "created_at": time.time(),
            "experiment": experiment,
            "instruction": instruction,
            "credentials_stored": False,
        },
    )
    return Lease(gateway, run_id, session_id, token, recorder, instruction), allocation


def write_endpoint_file(workspace: Path, endpoint: str) -> Path:
    """Publish only the local lease URL to an Agent Workspace."""

    path = workspace / ".libero-endpoint.json"
    atomic_json(path, {"schema_version": 1, "endpoint": endpoint})
    return path
