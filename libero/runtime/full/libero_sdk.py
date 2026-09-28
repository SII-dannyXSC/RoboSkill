#!/usr/bin/env python3
"""Agent-side SDK for the local LIBERO lease HTTP endpoint."""

from __future__ import annotations

import json
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


def load_depth(
    descriptor: dict[str, Any],
    *,
    root: str | Path | None = None,
) -> Any:
    """Load a persisted depth descriptor as a float32 NumPy array in meters.

    Both the current 16-bit PNG representation and historical raw float32
    artifacts are supported. ``root`` is only needed when ``path`` is relative.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - agent environment failure
        raise RuntimeError("loading LIBERO depth requires numpy") from exc

    if not isinstance(descriptor, dict):
        raise TypeError("depth descriptor must be a mapping")
    path_value = descriptor.get("path")
    shape = descriptor.get("shape")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("depth descriptor is missing path")
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(size) is not int or size < 1 for size in shape)
    ):
        raise ValueError("depth descriptor has an invalid shape")

    path = Path(path_value)
    if not path.is_absolute():
        path = (Path(root) if root is not None else Path.cwd()) / path

    encoding = descriptor.get("encoding")
    if encoding == "png-u16-linear":
        if len(shape) != 2:
            raise ValueError("PNG depth descriptor must have a 2-D shape")
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - agent environment failure
            raise RuntimeError("loading PNG depth requires Pillow") from exc
        scale = descriptor.get("scale_m_per_unit")
        offset = descriptor.get("offset_m", 0.0)
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError("PNG depth descriptor has an invalid scale_m_per_unit")
        if not isinstance(offset, (int, float)):
            raise ValueError("PNG depth descriptor has an invalid offset_m")
        with Image.open(path) as image:
            quantized = np.asarray(image).copy()
        if list(quantized.shape) != shape:
            raise ValueError(
                f"PNG depth shape {list(quantized.shape)} does not match descriptor {shape}"
            )
        return quantized.astype(np.float32) * np.float32(scale) + np.float32(offset)

    # Historical descriptors have no explicit encoding and use dtype=float32.
    if encoding in {None, "raw-float32"} and descriptor.get("dtype") == "float32":
        values = np.fromfile(path, dtype="<f4")
        expected = 1
        for size in shape:
            expected *= size
        if values.size != expected:
            raise ValueError(
                f"raw depth contains {values.size} values, expected {expected} for {shape}"
            )
        return values.reshape(tuple(shape))

    raise ValueError(f"unsupported depth encoding: {encoding!r}")


class LiberoSDKError(RuntimeError):
    """The local lease endpoint rejected an Agent request."""

    def __init__(self, status: int, payload: dict[str, Any]):
        super().__init__(f"local LIBERO endpoint HTTP {status}: {payload}")
        self.status = status
        self.payload = payload


@dataclass(frozen=True)
class MotionResult:
    """Outcome of feedback-driven ordinary Step requests."""

    reason: str
    observation: dict[str, Any]
    state: dict[str, Any] | None
    steps: int
    progress_m: float
    cross_track_error_m: float
    terminated: bool = False
    truncated: bool = False


class LiberoClient:
    """Small modifiable HTTP client that never receives the remote Token."""

    def __init__(self, endpoint: str, *, timeout: float = 3600.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        # This capability URL is always loopback. Explicitly bypass inherited
        # proxy settings rather than relying on NO_PROXY being configured.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def from_workspace(cls, workspace: str | Path = ".", *, timeout: float = 3600.0) -> "LiberoClient":
        """Load the local capability URL from a prepared Workspace."""

        path = Path(workspace).resolve() / ".libero-endpoint.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        endpoint = value.get("endpoint") if isinstance(value, dict) else None
        if not isinstance(endpoint, str) or not endpoint.startswith("http://127.0.0.1:"):
            raise RuntimeError(f"invalid local LIBERO endpoint file: {path}")
        return cls(endpoint, timeout=timeout)

    def _request(
        self,
        method: str,
        operation: str,
        *,
        body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers = {"Accept": "application/json"}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        delay = 0.25
        while True:
            request = urllib.request.Request(
                f"{self.endpoint}/{operation}",
                data=encoded,
                headers=headers,
                method=method,
            )
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    value = json.loads(response.read().decode("utf-8"))
                    if not isinstance(value, dict):
                        raise RuntimeError("local LIBERO endpoint returned non-object JSON")
                    return value
            except urllib.error.HTTPError as error:
                try:
                    value = json.loads(error.read().decode("utf-8"))
                except Exception:
                    value = {"error": {"code": "INVALID_ERROR_RESPONSE"}}
                if error.code not in {502, 503, 504}:
                    raise LiberoSDKError(error.code, value) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                pass
            time.sleep(delay)
            delay = min(delay * 2, 10.0)

    def reset(self, *, idempotency_key: str | None = None) -> dict[str, Any]:
        """Reset the existing remote Session without changing its seed."""

        key = idempotency_key or f"reset-{secrets.token_urlsafe(20)}"
        return self._request("POST", "reset", body={}, idempotency_key=key)

    def step(
        self,
        action: Iterable[float],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Execute one action through the trusted serial lease."""

        body: dict[str, Any] = {"action": list(action)}
        key = idempotency_key or f"step-{secrets.token_urlsafe(20)}"
        return self._request("POST", "step", body=body, idempotency_key=key)

    def move_until(
        self,
        initial_observation: dict[str, Any],
        *,
        direction: Sequence[float],
        max_distance_m: float,
        command_magnitude: float,
        stop_when: Callable[[dict[str, Any]], bool],
        on_step: Callable[[dict[str, Any], tuple[float, ...]], None] | None = None,
        max_steps: int = 120,
        distance_tolerance_m: float = 0.0005,
        stall_window: int = 10,
        min_window_progress_m: float = 0.0001,
        max_cross_track_error_m: float | None = None,
        gripper: float = 0.0,
        translation_scale_m: float = 0.05,
        settle_steps: int = 0,
    ) -> MotionResult:
        """Move along a world-frame line using measured EEF feedback.

        The normalized action is an OSC target offset, not achieved motion.
        Distance, path correction, and stall decisions use ``robot0_eef_pos``
        from successful Step responses. The trusted lease persists every Step
        before this method receives it; ``on_step`` is only for optional caller
        processing and is invoked once per successful internal Step.

        Use ``settle_steps=10`` for the first motion immediately after Reset.
        ``stalled`` reports insufficient progress under the current command; it
        does not prove that the requested position is globally unreachable.
        """

        unit_direction = _normalized_direction(direction)
        _validate_motion_parameters(
            max_distance_m=max_distance_m,
            command_magnitude=command_magnitude,
            max_steps=max_steps,
            distance_tolerance_m=distance_tolerance_m,
            stall_window=stall_window,
            min_window_progress_m=min_window_progress_m,
            max_cross_track_error_m=max_cross_track_error_m,
            gripper=gripper,
            translation_scale_m=translation_scale_m,
            settle_steps=settle_steps,
        )
        latest_observation = _observation(initial_observation)
        latest_state: dict[str, Any] | None = None

        if stop_when(latest_observation):
            return MotionResult(
                reason="condition_met",
                observation=latest_observation,
                state=None,
                steps=0,
                progress_m=0.0,
                cross_track_error_m=0.0,
            )

        completed_settle_steps = 0
        settle_action = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper)
        for settle_index in range(1, settle_steps + 1):
            latest_state = self.step(settle_action)
            latest_observation = _observation(latest_state)
            completed_settle_steps = settle_index
            if on_step is not None:
                on_step(latest_state, settle_action)
            terminal_result = _terminal_motion_result(
                latest_state,
                latest_observation,
                steps=completed_settle_steps,
                progress_m=0.0,
                cross_track_error_m=0.0,
            )
            if terminal_result is not None:
                return terminal_result
            if stop_when(latest_observation):
                return MotionResult(
                    reason="condition_met",
                    observation=latest_observation,
                    state=latest_state,
                    steps=completed_settle_steps,
                    progress_m=0.0,
                    cross_track_error_m=0.0,
                )

        start_pos = _eef_position(latest_observation)
        path_progress_history = [0.0]

        for motion_step in range(1, max_steps + 1):
            current_pos = _eef_position(latest_observation)
            progress, cross_track_error = _motion_progress(
                start_pos, current_pos, unit_direction
            )
            remaining = max_distance_m - progress
            if remaining <= distance_tolerance_m:
                return MotionResult(
                    reason="distance_limit",
                    observation=latest_observation,
                    state=latest_state,
                    steps=completed_settle_steps + motion_step - 1,
                    progress_m=progress,
                    cross_track_error_m=cross_track_error,
                )

            # Target a point ahead on the original line. Sending zero lateral
            # deltas would accept accumulated drift as the next OSC goal.
            lookahead_m = command_magnitude * translation_scale_m
            target_progress = min(
                max(progress, 0.0) + lookahead_m,
                max_distance_m,
            )
            target_pos = tuple(
                start_pos[index] + target_progress * unit_direction[index]
                for index in range(3)
            )
            target_error = tuple(
                target_pos[index] - current_pos[index] for index in range(3)
            )
            normalized_error = tuple(
                value / translation_scale_m for value in target_error
            )
            normalized_error_norm = sqrt(
                sum(value * value for value in normalized_error)
            )
            action_scale = (
                command_magnitude / normalized_error_norm
                if normalized_error_norm > command_magnitude
                else 1.0
            )
            translation_action = tuple(
                value * action_scale for value in normalized_error
            )
            action = (
                translation_action[0],
                translation_action[1],
                translation_action[2],
                0.0,
                0.0,
                0.0,
                gripper,
            )
            latest_state = self.step(action)
            latest_observation = _observation(latest_state)
            if on_step is not None:
                on_step(latest_state, action)

            current_pos = _eef_position(latest_observation)
            progress, cross_track_error = _motion_progress(
                start_pos, current_pos, unit_direction
            )
            path_progress_history.append(progress - cross_track_error)
            total_steps = completed_settle_steps + motion_step

            terminal_result = _terminal_motion_result(
                latest_state,
                latest_observation,
                steps=total_steps,
                progress_m=progress,
                cross_track_error_m=cross_track_error,
            )
            if terminal_result is not None:
                return terminal_result
            if stop_when(latest_observation):
                return MotionResult(
                    reason="condition_met",
                    observation=latest_observation,
                    state=latest_state,
                    steps=total_steps,
                    progress_m=progress,
                    cross_track_error_m=cross_track_error,
                )
            if (
                max_cross_track_error_m is not None
                and cross_track_error > max_cross_track_error_m
            ):
                return MotionResult(
                    reason="cross_track_limit",
                    observation=latest_observation,
                    state=latest_state,
                    steps=total_steps,
                    progress_m=progress,
                    cross_track_error_m=cross_track_error,
                )
            if max_distance_m - progress <= distance_tolerance_m:
                return MotionResult(
                    reason="distance_limit",
                    observation=latest_observation,
                    state=latest_state,
                    steps=total_steps,
                    progress_m=progress,
                    cross_track_error_m=cross_track_error,
                )
            if (
                len(path_progress_history) > stall_window
                and path_progress_history[-1]
                - path_progress_history[-1 - stall_window]
                < min_window_progress_m
            ):
                return MotionResult(
                    reason="stalled",
                    observation=latest_observation,
                    state=latest_state,
                    steps=total_steps,
                    progress_m=progress,
                    cross_track_error_m=cross_track_error,
                )

        progress, cross_track_error = _motion_progress(
            start_pos,
            _eef_position(latest_observation),
            unit_direction,
        )
        return MotionResult(
            reason="step_limit",
            observation=latest_observation,
            state=latest_state,
            steps=completed_settle_steps + max_steps,
            progress_m=progress,
            cross_track_error_m=cross_track_error,
        )

    def result(self) -> dict[str, Any]:
        """Return the server-authoritative result."""

        return self._request("GET", "result")


def _observation(value: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value.get("proprioception"), dict):
        return value
    nested = value.get("observation")
    if isinstance(nested, dict) and isinstance(nested.get("proprioception"), dict):
        return nested
    raise ValueError("value does not contain a LIBERO Observation")


def _eef_position(observation: dict[str, Any]) -> tuple[float, float, float]:
    values = observation["proprioception"].get("robot0_eef_pos")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("Observation robot0_eef_pos must contain exactly 3 values")
    position = tuple(float(value) for value in values)
    if not all(isfinite(value) for value in position):
        raise ValueError("robot0_eef_pos must contain only finite values")
    return position  # type: ignore[return-value]


def _normalized_direction(direction: Sequence[float]) -> tuple[float, float, float]:
    if len(direction) != 3:
        raise ValueError("direction must contain exactly 3 values")
    values = tuple(float(value) for value in direction)
    if not all(isfinite(value) for value in values):
        raise ValueError("direction must contain only finite values")
    norm = sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _motion_progress(
    start: tuple[float, float, float],
    current: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> tuple[float, float]:
    displacement = tuple(current[index] - start[index] for index in range(3))
    progress = sum(displacement[index] * direction[index] for index in range(3))
    lateral = tuple(
        displacement[index] - progress * direction[index] for index in range(3)
    )
    return progress, sqrt(sum(value * value for value in lateral))


def _terminal_motion_result(
    state: dict[str, Any],
    observation: dict[str, Any],
    *,
    steps: int,
    progress_m: float,
    cross_track_error_m: float,
) -> MotionResult | None:
    if state.get("terminated") is True:
        return MotionResult(
            reason="episode_terminated",
            observation=observation,
            state=state,
            steps=steps,
            progress_m=progress_m,
            cross_track_error_m=cross_track_error_m,
            terminated=True,
        )
    if state.get("truncated") is True:
        return MotionResult(
            reason="episode_truncated",
            observation=observation,
            state=state,
            steps=steps,
            progress_m=progress_m,
            cross_track_error_m=cross_track_error_m,
            truncated=True,
        )
    return None


def _validate_motion_parameters(
    *,
    max_distance_m: float,
    command_magnitude: float,
    max_steps: int,
    distance_tolerance_m: float,
    stall_window: int,
    min_window_progress_m: float,
    max_cross_track_error_m: float | None,
    gripper: float,
    translation_scale_m: float,
    settle_steps: int,
) -> None:
    finite_values = {
        "max_distance_m": max_distance_m,
        "command_magnitude": command_magnitude,
        "distance_tolerance_m": distance_tolerance_m,
        "min_window_progress_m": min_window_progress_m,
        "gripper": gripper,
        "translation_scale_m": translation_scale_m,
    }
    if max_cross_track_error_m is not None:
        finite_values["max_cross_track_error_m"] = max_cross_track_error_m
    for name, value in finite_values.items():
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")
    if not 0 < command_magnitude <= 1:
        raise ValueError("command_magnitude must be in (0, 1]")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not 0 <= distance_tolerance_m < max_distance_m:
        raise ValueError(
            "distance_tolerance_m must be non-negative and less than max_distance_m"
        )
    if stall_window < 1:
        raise ValueError("stall_window must be at least 1")
    if min_window_progress_m < 0:
        raise ValueError("min_window_progress_m must be non-negative")
    if max_cross_track_error_m is not None and max_cross_track_error_m <= 0:
        raise ValueError("max_cross_track_error_m must be positive when set")
    if not -1 <= gripper <= 1:
        raise ValueError("gripper must be in [-1, 1]")
    if translation_scale_m <= 0:
        raise ValueError("translation_scale_m must be positive")
    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")
