"""Dependency-free Python client for the LIBERO Agent Gateway."""

from __future__ import annotations

import base64
import json
import os
import secrets
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


class LiberoAPIError(RuntimeError):
    def __init__(self, status: int, code: str, request_id: str):
        super().__init__(f"LIBERO API error: status={status} code={code} request_id={request_id}")
        self.status = status
        self.code = code
        self.request_id = request_id


class LiberoCreateTransportError(RuntimeError):
    def __init__(self, idempotency_key: str):
        super().__init__(
            "Session create response was lost; retry create with the same "
            f"idempotency_key={idempotency_key!r}"
        )
        self.idempotency_key = idempotency_key


class LiberoSessionStartupError(RuntimeError):
    def __init__(self, session_id: str, code: str):
        super().__init__(f"LIBERO session {session_id} startup failed: {code}")
        self.session_id = session_id
        self.code = code


class LiberoEvaluationRunRequired(RuntimeError):
    def __init__(self):
        super().__init__(
            "LIBERO_RUN_ID is missing; start the agent with start_evaluation.py"
        )


@dataclass
class Observation:
    images_jpeg: Dict[str, bytes]
    proprioception: Dict[str, list]
    frame_id: int
    step_index: int
    annotations: Optional[Dict[str, Any]] = None
    depth_float32: Optional[Dict[str, Dict[str, Any]]] = None
    camera_calibration: Optional[Dict[str, Any]] = None

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "Observation":
        decoded_depth = {}
        for name, depth in payload.get("depth", {}).items():
            decoded_depth[name] = {
                "shape": depth["shape"],
                "unit": depth["unit"],
                "little_endian_float32_bytes": zlib.decompress(
                    base64.b64decode(depth["base64"], validate=True)
                ),
            }
        return cls(
            images_jpeg={
                name: base64.b64decode(image["base64"], validate=True)
                for name, image in payload["images"].items()
            },
            proprioception=payload["proprioception"],
            frame_id=payload["frame_id"],
            step_index=payload["step_index"],
            annotations=payload.get("annotations"),
            depth_float32=decoded_depth or None,
            camera_calibration=payload.get("camera_calibration"),
        )


class LiberoClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: float = 30.0,
        create_wait_timeout: float = 240.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.create_wait_timeout = create_wait_timeout

    def create_session(self, benchmark: str) -> Dict[str, Any]:
        """Deprecated v1 create; migrate to create_configured_session."""
        return self._request(
            "POST",
            "/v1/sessions",
            {"benchmark": benchmark},
            timeout=max(self.timeout, self.create_wait_timeout),
        )

    def capabilities(self) -> Dict[str, Any]:
        return self._request("GET", "/v2/capabilities")

    def tasks(self, benchmark: str) -> Dict[str, Any]:
        return self._request("GET", f"/v2/tasks/{benchmark}")

    def start_evaluation_run(
        self,
        benchmark: str,
        *,
        task_id: Optional[int] = None,
        episode_length: Optional[int] = None,
        observation_level: int = 1,
        bbox_scope: str = "initial",
        max_attempts: int = 100,
        label: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "task": {"benchmark": benchmark},
            "observation": {
                "level": observation_level,
                "bbox_scope": bbox_scope,
            },
            "max_attempts": max_attempts,
        }
        if task_id is not None:
            body["task"]["task_id"] = task_id
        if episode_length is not None:
            body["episode_length"] = episode_length
        if label is not None:
            body["label"] = label
        key = idempotency_key or secrets.token_urlsafe(24)
        for attempt in range(3):
            try:
                return self._request(
                    "POST",
                    "/v2/runs",
                    body,
                    headers={"Idempotency-Key": key},
                )
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise LiberoCreateTransportError(key) from None
                time.sleep(0.5 * (2**attempt))
        raise AssertionError("unreachable")

    def evaluation_run_status(self, run_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/v2/runs/{run_id}")

    def finish_evaluation_run(self, run_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/v2/runs/{run_id}/finish", {})

    def create_configured_session(
        self,
        benchmark: str,
        *,
        task_id: Optional[int] = None,
        episode_length: Optional[int] = None,
        observation_level: int = 1,
        bbox_scope: str = "initial",
        seed: Optional[int] = None,
        run_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        wait: bool = True,
    ) -> Dict[str, Any]:
        effective_run_id = run_id or os.environ.get("LIBERO_RUN_ID")
        if not effective_run_id:
            raise LiberoEvaluationRunRequired()
        body: Dict[str, Any] = {
            "run_id": effective_run_id,
            "task": {"benchmark": benchmark},
            "observation": {
                "level": observation_level,
                "bbox_scope": bbox_scope,
            },
        }
        if task_id is not None:
            body["task"]["task_id"] = task_id
        if episode_length is not None:
            body["episode_length"] = episode_length
        if seed is not None:
            body["seed"] = seed
        key = idempotency_key or secrets.token_urlsafe(24)
        created = None
        for attempt in range(3):
            try:
                created = self._request(
                    "POST",
                    "/v2/sessions",
                    body,
                    headers={"Idempotency-Key": key},
                )
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise LiberoCreateTransportError(key) from None
                time.sleep(0.5 * (2**attempt))
        assert created is not None
        if wait:
            return self.wait_until_ready(created["session_id"])
        return created

    def session_status(self, session_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/v2/sessions/{session_id}")

    def wait_until_ready(
        self,
        session_id: str,
        *,
        timeout: Optional[float] = None,
        poll_interval: float = 0.5,
        cancel_on_timeout: bool = True,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + (
            self.create_wait_timeout if timeout is None else timeout
        )
        while True:
            try:
                status = self.session_status(session_id)
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() < deadline:
                    time.sleep(poll_interval)
                    continue
                if cancel_on_timeout:
                    self._best_effort_close(session_id)
                raise TimeoutError(
                    f"LIBERO session {session_id} status could not be read before timeout"
                ) from None
            if status["state"] == "ready":
                return status
            if status["state"] == "failed":
                code = str(status.get("error_code", "unknown"))
                self._best_effort_close(session_id)
                raise LiberoSessionStartupError(session_id, code)
            if time.monotonic() >= deadline:
                if cancel_on_timeout:
                    self._best_effort_close(session_id)
                raise TimeoutError(
                    f"LIBERO session {session_id} was not ready before timeout"
                )
            time.sleep(poll_interval)

    def _best_effort_close(self, session_id: str) -> None:
        try:
            self.close(session_id)
        except Exception:
            pass

    def reset(self, session_id: str, *, api_version: str = "v2") -> Observation:
        """Reset the current Session and return the new Episode's observation."""
        return self.reset_episode(session_id, api_version=api_version)["observation"]

    def reset_episode(
        self, session_id: str, *, api_version: str = "v2"
    ) -> Dict[str, Any]:
        """Start a clean Episode in the same Session.

        The Session process, GPU slot, task, hidden seed and Run stay unchanged.
        The returned episode_index increases on every successful reset.
        """
        payload = self._request(
            "POST", f"/{api_version}/sessions/{session_id}/reset", {}
        )
        payload["observation"] = Observation.from_json(payload["observation"])
        return payload

    def step(
        self,
        session_id: str,
        action: Iterable[float],
        *,
        api_version: str = "v2",
    ) -> Dict[str, Any]:
        payload = self._request(
            "POST",
            f"/{api_version}/sessions/{session_id}/step",
            {"action": list(action)},
        )
        payload["observation"] = Observation.from_json(payload["observation"])
        return payload

    def result(self, session_id: str, *, api_version: str = "v2") -> Dict[str, Any]:
        return self._request("GET", f"/{api_version}/sessions/{session_id}/result")

    def close(self, session_id: str, *, api_version: str = "v2") -> Dict[str, Any]:
        return self._request("DELETE", f"/{api_version}/sessions/{session_id}")

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                **(headers or {}),
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout if timeout is None else timeout
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                error = json.loads(exc.read().decode("utf-8"))["error"]
            except Exception:
                error = {"code": "INVALID_ERROR_RESPONSE", "request_id": "unknown"}
            raise LiberoAPIError(exc.code, error["code"], error["request_id"]) from None
