#!/usr/bin/env python3
"""Regression tests for the strict no-tactile/no-gripper boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from libero_runtime import (
    NO_TACTILE_PROFILE,
    NO_TACTILE_PROPRIOCEPTION_KEYS,
    ArtifactRecorder,
    assert_no_tactile_payload,
    create_lease,
    observation_profile,
    project_no_tactile_payload,
)
from prompt_context import build_experiment_context


def experiment() -> dict:
    return {
        "task": {"benchmark": "libero_10", "task_id": 0},
        "seed": 0,
        "episode_length": 1000,
        "observation": {
            "level": 4,
            "bbox_scope": "initial",
            "profile": NO_TACTILE_PROFILE,
        },
    }


class NoTactileProjectionTests(unittest.TestCase):
    def test_profile_is_fail_closed(self) -> None:
        self.assertEqual(observation_profile(experiment()), NO_TACTILE_PROFILE)
        invalid = experiment()
        del invalid["observation"]["profile"]
        with self.assertRaises(RuntimeError):
            observation_profile(invalid)

    def test_projection_uses_allowlist_and_does_not_mutate_source(self) -> None:
        payload = {
            "observation": {
                "proprioception": {
                    **{key: [1.0] for key in NO_TACTILE_PROPRIOCEPTION_KEYS},
                    "robot0_joint_commanded_torque": [2.0],
                    "robot0_eef_force": [3.0],
                    "robot0_eef_torque": [4.0],
                    "robot0_gripper_torque": [5.0],
                    "future_contact_sensor": [6.0],
                    "robot0_gripper_qpos": [7.0],
                    "robot0_gripper_qvel": [8.0],
                }
            },
            "observation_spec": {
                "optional_proprioception_keys": [
                    *sorted(NO_TACTILE_PROPRIOCEPTION_KEYS),
                    "robot0_eef_force",
                    "robot0_eef_torque",
                ]
            },
        }
        original = json.loads(json.dumps(payload))

        projected = project_no_tactile_payload(payload)

        self.assertEqual(payload, original)
        self.assertEqual(
            set(projected["observation"]["proprioception"]),
            set(NO_TACTILE_PROPRIOCEPTION_KEYS),
        )
        self.assertEqual(
            set(projected["observation_spec"]["optional_proprioception_keys"]),
            set(NO_TACTILE_PROPRIOCEPTION_KEYS),
        )
        self.assertEqual(projected["observation_spec"]["profile"], NO_TACTILE_PROFILE)
        assert_no_tactile_payload(projected)

    def test_leak_guard_rejects_future_tactile_key(self) -> None:
        for key in (
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
            "gripper_force",
            "jointTorque",
            "contact_wrench",
            "tactile_array",
        ):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                assert_no_tactile_payload({"observation": {key: [0.0]}})

    def test_manifest_records_projection_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ArtifactRecorder(root, {"experiment": experiment()})
            manifest = json.loads(
                (root / "libero-artifacts" / "manifest.json").read_text(encoding="utf-8")
            )
        projection = manifest["observation_projection"]
        self.assertEqual(projection["profile"], NO_TACTILE_PROFILE)
        self.assertEqual(
            set(projection["proprioception_allowlist"]),
            set(NO_TACTILE_PROPRIOCEPTION_KEYS),
        )

    def test_profile_is_not_sent_to_gateway(self) -> None:
        class FakeGateway:
            sent: dict | None = None

            def allocate(self, value: dict, _key: str) -> dict:
                self.sent = value
                return {
                    "run": {"run_id": "run-test"},
                    "session": {"session_id": "session-test"},
                    "agent_token": "secret",
                }

            def wait_ready(self, _session_id: str, _token: str) -> dict:
                return {"instruction": "Pick up the object"}

        source = experiment()
        gateway = FakeGateway()
        with tempfile.TemporaryDirectory() as directory:
            create_lease(gateway, Path(directory), source, allocation_key="test")
        self.assertIsNotNone(gateway.sent)
        assert gateway.sent is not None
        self.assertNotIn("profile", gateway.sent["observation"])
        self.assertEqual(source["observation"]["profile"], NO_TACTILE_PROFILE)

    def test_prompt_states_the_hidden_channels(self) -> None:
        context = build_experiment_context(
            experiment(), instruction="Pick up the object", workflow_variant="plain"
        )
        self.assertIn(NO_TACTILE_PROFILE, context)
        self.assertIn("robot0_eef_force", context)
        self.assertIn("robot0_gripper_", context)
        self.assertIn("不提供", context)


if __name__ == "__main__":
    unittest.main()
