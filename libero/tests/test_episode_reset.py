from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from libero_gateway.settings import Settings
from libero_gateway.worker_pool import ConflictError, SimulationManager


def settings_for(root: Path) -> Settings:
    agents = root / "agents.json"
    agents.write_text(
        json.dumps(
            {
                "agents": {
                    "legacy": {
                        "token_sha256": hashlib.sha256(b"unused").hexdigest(),
                        "allowed_benchmarks": ["libero_goal"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return Settings(
        backend="mock",
        agents_file=str(agents),
        audit_log=str(root / "audit.jsonl"),
        evaluations_db=str(root / "evaluations.sqlite3"),
        gpu_ids=(0,),
        max_sessions_per_gpu=2,
        image_height=8,
        image_width=8,
        default_episode_steps=2,
        max_episode_steps=2000,
        process_start_timeout_seconds=10,
        request_timeout_seconds=5,
    )


def wait_ready(manager: SimulationManager, session_id: str, owner: str) -> None:
    for _ in range(100):
        status = manager.session_status(session_id, owner)
        if status["state"] == "ready":
            return
        if status["state"] == "failed":
            raise AssertionError(status)
        time.sleep(0.02)
    raise AssertionError("session did not become ready")


def allocate(
    manager: SimulationManager,
    *,
    owner: str,
    key: str,
    episode_length: int,
):
    config = {
        "benchmark": "libero_goal",
        "task_id": 0,
        "episode_length": episode_length,
        "observation_level": 1,
        "bbox_scope": "initial",
    }
    run = manager.begin_run(
        owner,
        launcher_owner=owner,
        idempotency_key=key,
        request_fingerprint=key,
        config=config,
        max_attempts=1,
    )
    session = manager.begin_session(
        owner,
        "libero_goal",
        1,
        idempotency_key="session_" + key,
        request_fingerprint="session_" + key,
        run_id=run["run_id"],
        task_id=0,
        seed=0,
        episode_length=episode_length,
        observation_level=1,
        bbox_scope="initial",
    )
    wait_ready(manager, session["session_id"], owner)
    return run["run_id"], session["session_id"]


class EpisodeLifecycleTest(unittest.TestCase):
    """One evaluation Session is one terminal attempt.

    An Agent may reset early and reuse its Session, but a terminal success or
    truncation is durably recorded and closes that attempt. A later attempt must
    use a new Session under a Run whose ``max_attempts`` permits it.
    """

    def test_reset_after_truncation_is_denied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(settings_for(Path(directory)))
            try:
                owner = "truncate-owner"
                run_id, session_id = allocate(
                    manager,
                    owner=owner,
                    key="truncate-reset-test",
                    episode_length=2,
                )
                first = manager.reset(session_id, owner)
                self.assertEqual(first["episode_index"], 1)
                action = [0.0] * 7
                self.assertFalse(manager.step(session_id, owner, action)["truncated"])
                terminal = manager.step(session_id, owner, action)
                self.assertTrue(terminal["truncated"])
                self.assertEqual(
                    manager.result(session_id, owner),
                    {"status": "completed", "success": False, "steps": 2},
                )
                with self.assertRaises(ConflictError):
                    manager.reset(session_id, owner)
                run = manager.run_status(run_id, owner)
                self.assertEqual(run["episodes_completed"], 1)
                self.assertEqual(run["successful_episodes"], 0)
                self.assertEqual(run["active_sessions"], 1)
                manager.close_session(session_id, owner)
                manager.finish_run(run_id, owner)
            finally:
                manager.shutdown()

    def test_reset_after_success_is_denied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(settings_for(Path(directory)))
            try:
                owner = "success-owner"
                run_id, session_id = allocate(
                    manager,
                    owner=owner,
                    key="success-reset-denied",
                    episode_length=4,
                )
                manager.reset(session_id, owner)
                action = [0.0] * 7
                for _ in range(3):
                    terminal = manager.step(session_id, owner, action)
                self.assertTrue(terminal["terminated"])
                with self.assertRaises(ConflictError):
                    manager.reset(session_id, owner)
                run = manager.run_status(run_id, owner)
                self.assertEqual(run["episodes_completed"], 1)
                self.assertEqual(run["successful_episodes"], 1)
                self.assertEqual(run["active_sessions"], 1)
                manager.close_session(session_id, owner)
                manager.finish_run(run_id, owner)
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
