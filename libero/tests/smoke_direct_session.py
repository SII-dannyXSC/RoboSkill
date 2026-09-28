#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import secrets
import time
import urllib.error
import urllib.request


BASE_URL = os.environ.get("LIBERO_API_URL", "http://127.0.0.1:18080").rstrip("/")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(method: str, path: str, body=None, *, token=None, headers=None):
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        **(headers or {}),
    }
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path, data=data, headers=request_headers, method=method
    )
    with opener.open(req, timeout=180) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def close_enough(left, right, tolerance=1e-6):
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right)
    )


def bboxes_close(left, right, tolerance=1):
    if set(left) != set(right):
        return False
    for camera in left:
        left_boxes = sorted(left[camera], key=lambda item: item["label"])
        right_boxes = sorted(right[camera], key=lambda item: item["label"])
        if [item["label"] for item in left_boxes] != [
            item["label"] for item in right_boxes
        ]:
            return False
        for first, second in zip(left_boxes, right_boxes):
            if any(
                abs(int(a) - int(b)) > tolerance
                for a, b in zip(first["xyxy"], second["xyxy"])
            ):
                return False
    return True


def main() -> int:
    run_id = None
    session_id = None
    token = None
    try:
        _, allocation = request(
            "POST",
            "/v2/experiments",
            {
                "label": "deployment-smoke-direct-session",
                "task": {"benchmark": "libero_10", "task_id": 9},
                "episode_length": 2000,
                "observation": {"level": 3, "bbox_scope": "initial"},
                "seed": 0,
            },
            headers={
                "Idempotency-Key": "deploy_" + secrets.token_urlsafe(24)
            },
        )
        run_id = allocation["run"]["run_id"]
        session_id = allocation["session"]["session_id"]
        token = allocation["agent_token"]
        assert token.startswith("lbr_session_")
        assert allocation["session"]["session_config"]["seed"] == 0

        deadline = time.monotonic() + 240
        while True:
            _, status = request("GET", f"/v2/sessions/{session_id}", token=token)
            if status["state"] == "ready":
                break
            if status["state"] == "failed":
                raise RuntimeError(f"Session startup failed: {status}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Session did not become ready")
            time.sleep(0.5)

        _, catalog = request("GET", "/v2/tasks/libero_10", token=token)
        task = next(item for item in catalog["tasks"] if item["task_id"] == 9)
        assert set(task) == {"task_id", "instruction"}, task

        reset_key = "reset_" + secrets.token_urlsafe(24)
        _, first = request(
            "POST",
            f"/v2/sessions/{session_id}/reset",
            {},
            token=token,
            headers={"Idempotency-Key": reset_key},
        )
        _, replayed_reset = request(
            "POST",
            f"/v2/sessions/{session_id}/reset",
            {},
            token=token,
            headers={"Idempotency-Key": reset_key},
        )
        assert first == replayed_reset
        _, second = request(
            "POST",
            f"/v2/sessions/{session_id}/reset",
            {},
            token=token,
            headers={"Idempotency-Key": "reset_" + secrets.token_urlsafe(24)},
        )
        assert first["episode_index"] == 1, first["episode_index"]
        assert second["episode_index"] == 2, second["episode_index"]
        assert "phase_successes" not in first, first
        assert "phase_successes" not in second, second
        first_obs = first["observation"]
        second_obs = second["observation"]
        pose_keys = (
            "robot0_joint_pos",
            "robot0_gripper_qpos",
            "robot0_eef_pos",
            "robot0_eef_quat",
        )
        same_pose = all(
            close_enough(
                first_obs["proprioception"][key],
                second_obs["proprioception"][key],
            )
            for key in pose_keys
        )
        assert same_pose
        first_boxes = first_obs["annotations"]["object_bboxes"]
        second_boxes = second_obs["annotations"]["object_bboxes"]
        same_bboxes = bboxes_close(first_boxes, second_boxes)
        image_bytes_exact = first_obs["images"] == second_obs["images"]

        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        step_key = "step_" + secrets.token_urlsafe(24)
        _, first_step = request(
            "POST",
            f"/v2/sessions/{session_id}/step",
            {"action": action, "expected_step_index": 0},
            token=token,
            headers={"Idempotency-Key": step_key},
        )
        _, replayed_step = request(
            "POST",
            f"/v2/sessions/{session_id}/step",
            {"action": action, "expected_step_index": 0},
            token=token,
            headers={"Idempotency-Key": step_key},
        )
        assert first_step == replayed_step
        assert first_step["observation"]["step_index"] == 1
        assert "phase_successes" not in first_step, first_step

        _, result = request(
            "GET", f"/v2/sessions/{session_id}/result", token=token
        )
        assert "phase_successes" not in result, result
        _, progress = request(
            "GET",
            f"/v2/sessions/{session_id}/harness-progress",
            token=token,
        )
        assert progress["total_phases"] == 4, progress

        try:
            request(
                "POST",
                f"/v2/sessions/{session_id}/step",
                {"action": action, "expected_step_index": 0},
                token=token,
                headers={"Idempotency-Key": "step_" + secrets.token_urlsafe(24)},
            )
            raise AssertionError("Stale step index was accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code == 409, exc
            error = json.loads(exc.read().decode("utf-8"))["error"]
            assert error["code"] == "STEP_INDEX_MISMATCH", error

        try:
            request(
                "POST",
                "/v2/sessions",
                {
                    "run_id": run_id,
                    "task": {"benchmark": "libero_10", "task_id": 9},
                    "episode_length": 2000,
                    "observation": {"level": 3, "bbox_scope": "initial"},
                },
                token=token,
                headers={"Idempotency-Key": "must-be-denied"},
            )
            raise AssertionError("Session-bound token created a second Session")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403, exc

        print(
            "direct_session_smoke_ok "
            "fixed_seed=0 reset_episode_index=2 same_pose=true "
            "reset_replay_exact=true step_replay_exact=true "
            "public_interface_unchanged=true harness_progress=true "
            "stale_step_denied=true "
            f"bbox_pixels_exact={str(same_bboxes).lower()} "
            f"image_bytes_exact={str(image_bytes_exact).lower()} "
            "second_session_denied=true"
        )
        return 0
    finally:
        if token and session_id:
            try:
                request("DELETE", f"/v2/sessions/{session_id}", token=token)
            except Exception:
                pass
        if token and run_id:
            try:
                request("POST", f"/v2/runs/{run_id}/finish", {}, token=token)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
