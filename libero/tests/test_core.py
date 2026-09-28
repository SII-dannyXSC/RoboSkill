import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import numpy as np
try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # Production venv need not install test-only httpx.
    TestClient = None

from libero_gateway.auth import (
    RUN_MANAGE_SCOPE,
    SESSION_OPERATE_SCOPE,
    AuthenticationError,
    TokenStore,
    token_sha256,
)
from libero_gateway.app import create_app
from libero_gateway.backend import (
    EpisodeOptions,
    InvalidSessionState,
    LiberoEpisode,
    MockEpisode,
    validate_action,
)
from libero_gateway.observations import public_observation
from libero_gateway.observations import ObservationError
from libero_gateway.phases import (
    LIBERO10_PHASES,
    PhaseCondition,
    PhaseTracker,
    phases_for_task,
)
from libero_gateway.schemas import ResetResponse, ResultResponse, StepResponse
from libero_gateway.settings import Settings
from libero_gateway.evaluations import EvaluationStore
from libero_gateway.worker_pool import (
    CapacityError,
    EvaluationRunRequired,
    EvaluationPersistenceFailure,
    IdempotencyConflict,
    QuotaError,
    RunConfigMismatch,
    RunNotFound,
    SimulationManager,
)


class TokenStoreTest(unittest.TestCase):
    def test_authentication_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.json"
            path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "agent-a": {
                                "token_sha256": token_sha256("secret-a"),
                            "allowed_benchmarks": ["libero_goal"],
                            "max_sessions": 2,
                            "max_observation_level": 4,
                            "max_episode_steps": 900,
                            "allow_fixed_seed": True,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            store = TokenStore.load(str(path))
            identity = store.authenticate("secret-a")
            self.assertEqual(identity.agent_id, "agent-a")
            self.assertEqual(identity.owner_id, "agent-a")
            self.assertEqual(
                identity.scopes,
                frozenset({RUN_MANAGE_SCOPE, SESSION_OPERATE_SCOPE}),
            )
            self.assertEqual(identity.max_sessions, 2)
            self.assertEqual(identity.max_observation_level, 4)
            self.assertEqual(identity.max_episode_steps, 900)
            self.assertTrue(identity.allow_fixed_seed)
            with self.assertRaises(AuthenticationError):
                store.authenticate("secret-b")

    def test_split_credentials_share_owner_but_not_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.json"
            path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "trial-agent": {
                                "token_sha256": token_sha256("agent-secret"),
                                "owner_id": "trial",
                                "scopes": [SESSION_OPERATE_SCOPE],
                                "allowed_benchmarks": ["libero_10"],
                            },
                            "trial-launcher": {
                                "token_sha256": token_sha256("launcher-secret"),
                                "owner_id": "trial",
                                "scopes": [RUN_MANAGE_SCOPE],
                                "allowed_benchmarks": ["libero_10"],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            store = TokenStore.load(str(path))
            agent = store.authenticate("agent-secret")
            launcher = store.authenticate("launcher-secret")
            self.assertEqual(agent.owner_id, launcher.owner_id)
            self.assertNotIn(RUN_MANAGE_SCOPE, agent.scopes)
            self.assertNotIn(SESSION_OPERATE_SCOPE, launcher.scopes)

    def test_run_token_is_bound_rotated_and_revoked(self):
        store = TokenStore({})
        first = store.issue_run_token(
            run_id="run_one",
            owner_id="qwen::level-2",
            benchmark="libero_10",
            observation_level=2,
            episode_steps=2000,
            max_sessions=8,
            ttl_seconds=60,
        )
        identity = store.authenticate(first)
        self.assertEqual(identity.owner_id, "qwen::level-2")
        self.assertEqual(identity.bound_run_id, "run_one")
        self.assertEqual(identity.scopes, frozenset({SESSION_OPERATE_SCOPE}))
        self.assertNotIn(RUN_MANAGE_SCOPE, identity.scopes)

        second = store.issue_run_token(
            run_id="run_one",
            owner_id="qwen::level-2",
            benchmark="libero_10",
            observation_level=2,
            episode_steps=2000,
            max_sessions=8,
            ttl_seconds=60,
        )
        with self.assertRaises(AuthenticationError):
            store.authenticate(first)
        self.assertEqual(store.authenticate(second).bound_run_id, "run_one")
        store.revoke_run_token("run_one")
        with self.assertRaises(AuthenticationError):
            store.authenticate(second)


@unittest.skipIf(TestClient is None, "httpx is not installed")
class DynamicRunApiTest(unittest.TestCase):
    def test_launcher_gets_one_run_token_per_model_level_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.json"
            launcher_token = "launcher-secret"
            agents.write_text(
                json.dumps(
                    {
                        "agents": {
                            "qwen-launcher": {
                                "token_sha256": token_sha256(launcher_token),
                                "owner_id": "qwen",
                                "scopes": [RUN_MANAGE_SCOPE],
                                "allowed_benchmarks": ["libero_goal"],
                                "max_sessions": 8,
                                "max_observation_level": 4,
                                "max_episode_steps": 2000,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            settings = Settings(
                agents_file=str(agents),
                audit_log=str(root / "audit.jsonl"),
                evaluations_db=str(root / "evaluations.sqlite3"),
                default_episode_steps=2000,
                max_episode_steps=2000,
            )
            headers = {"Authorization": f"Bearer {launcher_token}"}
            with TestClient(create_app(settings)) as client:
                responses = []
                for level in (1, 2):
                    response = client.post(
                        "/v2/runs",
                        headers={**headers, "Idempotency-Key": f"api-level-{level}"},
                        json={
                            "task": {"benchmark": "libero_goal", "task_id": 0},
                            "episode_length": 2000,
                            "observation": {"level": level},
                            "max_attempts": 1,
                        },
                    )
                    self.assertEqual(response.status_code, 201, response.text)
                    self.assertEqual(response.headers["cache-control"], "no-store")
                    responses.append(response.json())

                self.assertTrue(responses[0]["agent_identity"].startswith("visitor_"))
                self.assertTrue(responses[1]["agent_identity"].startswith("visitor_"))
                self.assertNotEqual(
                    responses[0]["agent_identity"], responses[1]["agent_identity"]
                )
                self.assertNotEqual(responses[0]["agent_token"], responses[1]["agent_token"])

                retried = client.post(
                    "/v2/runs",
                    headers={**headers, "Idempotency-Key": "api-level-2"},
                    json={
                        "task": {"benchmark": "libero_goal", "task_id": 0},
                        "episode_length": 2000,
                        "observation": {"level": 2},
                        "max_attempts": 1,
                    },
                )
                self.assertEqual(retried.status_code, 201, retried.text)
                self.assertEqual(retried.json()["run_id"], responses[1]["run_id"])
                self.assertEqual(
                    retried.json()["agent_identity"], responses[1]["agent_identity"]
                )
                self.assertNotEqual(
                    retried.json()["agent_token"], responses[1]["agent_token"]
                )

                dynamic_headers = {
                    "Authorization": f"Bearer {responses[0]['agent_token']}"
                }
                capability = client.get("/v2/capabilities", headers=dynamic_headers)
                self.assertEqual(capability.status_code, 200)
                self.assertEqual(
                    capability.json()["observation_levels"]["maximum"], 1
                )
                forbidden = client.get(
                    f"/v2/runs/{responses[0]['run_id']}", headers=dynamic_headers
                )
                self.assertEqual(forbidden.status_code, 403)

                finished = client.post(
                    f"/v2/runs/{responses[0]['run_id']}/finish", headers=headers
                )
                self.assertEqual(finished.status_code, 200, finished.text)
                revoked = client.get("/v2/capabilities", headers=dynamic_headers)
                self.assertEqual(revoked.status_code, 401)


class ObservationTest(unittest.TestCase):
    def test_privileged_fields_are_not_serialized(self):
        raw = {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_joint_pos": np.zeros(7),
            "robot0_gripper_qpos": np.zeros(2),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "object_positions": {"secret": [1, 2, 3]},
            "camera_intrinsics": np.eye(3),
            "camera_extrinsics": np.eye(4),
            "mujoco_state": np.arange(10),
        }
        public = public_observation(raw, quality=80, frame_id=0, step_index=0)
        serialized = json.dumps(public)
        self.assertNotIn("object_positions", serialized)
        self.assertNotIn("camera_intrinsics", serialized)
        self.assertNotIn("camera_extrinsics", serialized)
        self.assertNotIn("mujoco_state", serialized)
        self.assertIn("agentview_rgb", public["images"])
        self.assertIn("robot0_joint_pos", public["proprioception"])

    def test_level_3_adds_extended_proprioception_without_depth(self):
        raw = {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_joint_pos": np.zeros(7),
            "robot0_gripper_qpos": np.zeros(2),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_joint_vel": np.ones(7),
            "robot0_joint_commanded_torque": np.ones(7),
        }
        public = public_observation(
            raw,
            quality=80,
            frame_id=0,
            step_index=0,
            level=3,
            object_bboxes={"agentview_rgb": [], "wrist_rgb": []},
        )
        self.assertIn("annotations", public)
        self.assertIn("robot0_joint_vel", public["proprioception"])
        self.assertIn("robot0_joint_commanded_torque", public["proprioception"])
        self.assertNotIn("depth", public)
        self.assertNotIn("camera_calibration", public)

    def test_level_4_adds_depth_and_camera_calibration(self):
        raw = {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_joint_pos": np.zeros(7),
            "robot0_gripper_qpos": np.zeros(2),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_joint_vel": np.ones(7),
            "robot0_joint_commanded_torque": np.ones(7),
        }
        calibration = {
            name: {
                "intrinsic": np.eye(3).tolist(),
                "camera_to_world": np.eye(4).tolist(),
                "convention": "opencv",
            }
            for name in ("agentview_rgb", "wrist_rgb")
        }
        public = public_observation(
            raw,
            quality=80,
            frame_id=0,
            step_index=0,
            level=4,
            depth_maps={
                "agentview_depth": np.ones((8, 8)),
                "wrist_depth": np.ones((8, 8)),
            },
            camera_calibration=calibration,
            object_bboxes={"agentview_rgb": [], "wrist_rgb": []},
        )
        self.assertIn("annotations", public)
        self.assertIn("depth", public)
        self.assertIn("camera_calibration", public)
        self.assertIn("robot0_joint_vel", public["proprioception"])
        self.assertIn("robot0_joint_commanded_torque", public["proprioception"])
        self.assertNotIn("object_positions", json.dumps(public))

    def test_required_proprioception_cannot_silently_disappear(self):
        raw = {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        with self.assertRaises(ObservationError):
            public_observation(raw, quality=80, frame_id=0, step_index=0)


class ActionTest(unittest.TestCase):
    def test_action_validation(self):
        self.assertEqual(validate_action([0] * 7), (0.0,) * 7)
        for invalid in ([0] * 6, [0] * 6 + [2], [0] * 6 + [float("nan")]):
            with self.assertRaises(ValueError):
                validate_action(invalid)


class MockEpisodeTest(unittest.TestCase):
    def test_episode_lifecycle(self):
        settings = Settings(image_height=8, image_width=8, max_episode_steps=5)
        episode = MockEpisode(settings, "libero_goal", 0, 1, 0)
        reset = episode.reset()
        self.assertFalse(reset["terminated"])
        self.assertFalse(episode.step((0.0,) * 7)["terminated"])
        self.assertFalse(episode.step((0.0,) * 7)["terminated"])
        self.assertTrue(episode.step((0.0,) * 7)["terminated"])
        self.assertTrue(episode.result()["success"])
        with self.assertRaises(InvalidSessionState):
            episode.step((0.0,) * 7)
        restarted = episode.reset()
        self.assertFalse(restarted["terminated"])
        self.assertEqual(restarted["observation"]["step_index"], 0)
        self.assertFalse(episode.step((0.0,) * 7)["terminated"])

    def test_client_episode_length_and_bbox_scope(self):
        settings = Settings(image_height=8, image_width=8)
        episode = MockEpisode(
            settings,
            "libero_goal",
            0,
            1,
            0,
            EpisodeOptions(
                episode_length=1, observation_level=2, bbox_scope="initial"
            ),
        )
        reset = episode.reset()
        self.assertIn("annotations", reset["observation"])
        step = episode.step((0.0,) * 7)
        self.assertTrue(step["truncated"])
        self.assertNotIn("annotations", step["observation"])

    def test_libero10_tracks_phases_without_changing_public_responses(self):
        settings = Settings(image_height=8, image_width=8, max_episode_steps=5)
        episode = MockEpisode(settings, "libero_10", 5, 1, 0)

        reset = episode.reset()
        phases = reset.pop("_phase_successes")
        ResetResponse.model_validate({**reset, "episode_index": 1})
        self.assertEqual(len(phases), 4)
        self.assertFalse(any(item["success"] for item in phases))

        for expected_count in (1, 2, 4):
            response = episode.step((0.0,) * 7)
            phases = response.pop("_phase_successes")
            response.pop("_success_monotonic_ns", None)
            StepResponse.model_validate(response)
            self.assertEqual(
                sum(item["success"] for item in phases),
                expected_count,
            )

        self.assertTrue(response["terminated"])
        result = episode.result()
        phases = result.pop("_phase_successes")
        ResultResponse.model_validate(result)
        self.assertTrue(all(item["success"] for item in phases))


class PhaseDefinitionTest(unittest.TestCase):
    def test_every_libero10_task_has_four_numbered_phases(self):
        self.assertEqual(set(LIBERO10_PHASES), set(range(10)))
        for task_id, phases in LIBERO10_PHASES.items():
            self.assertGreaterEqual(len(phases), 4, task_id)
            self.assertEqual(
                [phase.phase_id for phase in phases],
                list(range(1, len(phases) + 1)),
            )

    def test_other_benchmarks_do_not_define_libero10_phases(self):
        self.assertEqual(phases_for_task("libero_goal", 0), ())

    def test_tracker_records_independent_conditions_and_keeps_achievements(self):
        phases = LIBERO10_PHASES[9]
        tracker = PhaseTracker(phases)
        satisfied = {"predicate", "success"}

        tracker.update(7, lambda phase: phase.condition.kind in satisfied)
        statuses = tracker.statuses()
        self.assertFalse(statuses[0]["success"])
        self.assertTrue(all(item["success"] for item in statuses[1:]))

        satisfied.add("grasped")
        tracker.update(8, lambda phase: phase.condition.kind in satisfied)
        statuses = tracker.statuses()
        self.assertTrue(all(item["success"] for item in statuses))
        self.assertEqual(
            [item["first_achieved_step"] for item in statuses],
            [8, 7, 7, 7],
        )

        tracker.update(9, lambda _phase: False)
        self.assertTrue(all(item["success"] for item in tracker.statuses()))

    def test_libero_phase_conditions_use_simulator_state(self):
        episode = object.__new__(LiberoEpisode)
        task_object = SimpleNamespace(
            root_body="black_book_body", contact_geoms=("book_geom",)
        )
        predicate_calls = []
        inner = SimpleNamespace(
            get_object=lambda name: task_object,
            robots=[SimpleNamespace(gripper="gripper")],
            _check_grasp=lambda gripper, geoms: (
                gripper == "gripper" and geoms == ("book_geom",)
            ),
            _eval_predicate=lambda state: predicate_calls.append(state) or True,
        )
        simulation = SimpleNamespace(
            model=SimpleNamespace(body_name2id=lambda name: 0),
            data=SimpleNamespace(body_xpos=np.asarray([[0.0, 0.0, 1.03]])),
        )
        episode._env = SimpleNamespace(env=inner, sim=simulation)
        episode._initial_object_heights = {"black_book_1": 1.0}
        episode._latest_eef_position = np.asarray([0.0, 0.0, 1.0])
        episode._success = True

        self.assertTrue(
            episode._phase_condition_satisfied(
                PhaseCondition("grasped", ("black_book_1",))
            )
        )
        self.assertTrue(
            episode._phase_condition_satisfied(
                PhaseCondition("lifted", ("black_book_1",))
            )
        )
        self.assertTrue(
            episode._phase_condition_satisfied(
                PhaseCondition("near", ("black_book_1",))
            )
        )
        predicate = ("in", "black_book_1", "desk_caddy_1_back_contain_region")
        self.assertTrue(
            episode._phase_condition_satisfied(
                PhaseCondition("predicate", predicate)
            )
        )
        self.assertEqual(predicate_calls, [predicate])
        self.assertTrue(
            episode._phase_condition_satisfied(PhaseCondition("success"))
        )


class SettingsTest(unittest.TestCase):
    def test_fixed_task_configuration(self):
        settings = Settings(fixed_benchmark="libero_goal", fixed_task_id=1)
        with tempfile.TemporaryDirectory() as directory:
            agents = Path(directory) / "agents.json"
            agents.write_text('{"agents":{"a":{"token_sha256":"' + "0" * 64 + '","allowed_benchmarks":["libero_goal"]}}}')
            object.__setattr__(settings, "agents_file", str(agents))
            settings.validate()

    def test_fixed_task_requires_benchmark(self):
        settings = Settings(fixed_task_id=1)
        with self.assertRaises(ValueError):
            settings.validate()


class DynamicSessionManagerTest(unittest.TestCase):
    def _settings(self, directory, **overrides):
        values = dict(
            backend="mock",
            gpu_ids=(0,),
            max_sessions_per_gpu=1,
            process_start_timeout_seconds=10,
            request_timeout_seconds=5,
            session_idle_seconds=60,
            image_height=8,
            image_width=8,
            audit_log=str(Path(directory) / "audit.jsonl"),
            evaluations_db=str(Path(directory) / "evaluations.sqlite3"),
        )
        values.update(overrides)
        return Settings(**values)

    def test_processes_follow_session_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            try:
                self.assertEqual(manager.health()["workers"], 0)
                created = manager.create_session(
                    "agent-a",
                    "libero_goal",
                    1,
                    task_id=0,
                    episode_length=1,
                    observation_level=4,
                )
                self.assertEqual(manager.health()["workers"], 1)
                reset = manager.reset(created["session_id"], "agent-a")
                self.assertIn("depth", reset["observation"])
                step = manager.step(created["session_id"], "agent-a", [0] * 7)
                self.assertTrue(step["truncated"])
                manager.close_session(created["session_id"], "agent-a")
                self.assertEqual(manager.health()["workers"], 0)
                self.assertEqual(manager.health()["slots"]["free"], 1)
            finally:
                manager.shutdown()

    def test_harness_reads_progress_only_from_finalization_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            try:
                created = manager.create_session(
                    "agent-a",
                    "libero_10",
                    1,
                    task_id=0,
                    episode_length=4,
                )
                session_id = created["session_id"]
                reset = manager.reset(session_id, "agent-a")
                self.assertNotIn("phase_successes", reset)
                self.assertNotIn("_phase_successes", reset)
                self.assertEqual(
                    manager.task_progress(session_id, "agent-a")["completed_phases"],
                    0,
                )

                for expected_count in (1, 2, 4):
                    response = manager.step(session_id, "agent-a", [0] * 7)
                    self.assertNotIn("phase_successes", response)
                    self.assertNotIn("_phase_successes", response)
                    progress = manager.task_progress(session_id, "agent-a")
                    self.assertEqual(
                        progress["completed_phases"],
                        expected_count,
                    )

                result = manager.result(session_id, "agent-a")
                self.assertTrue(result["success"])
                self.assertNotIn("phase_successes", result)
                progress = manager.task_progress(session_id, "agent-a")
                self.assertEqual(progress["total_phases"], 4)
                self.assertEqual(progress["completion_ratio"], 1.0)
                self.assertTrue(all(phase["success"] for phase in progress["phases"]))
                manager.close_session(session_id, "agent-a")
            finally:
                manager.shutdown()

    def test_harness_progress_survives_early_disconnect(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            try:
                created = manager.create_session(
                    "agent-a",
                    "libero_10",
                    1,
                    task_id=0,
                    episode_length=20,
                )
                session_id = created["session_id"]
                manager.reset(session_id, "agent-a")
                manager.step(session_id, "agent-a", [0] * 7)
                manager.step(session_id, "agent-a", [0] * 7)

                progress = manager.task_progress(session_id, "agent-a")
                self.assertEqual(progress["completed_phases"], 2)
                self.assertEqual(progress["completion_ratio"], 0.5)
                manager.close_session(session_id, "agent-a")

                events = [
                    json.loads(line)
                    for line in (Path(directory) / "audit.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                closed = [event for event in events if event["event"] == "session_closed"][-1]
                self.assertEqual(closed["task_progress"], progress)
            finally:
                manager.shutdown()

    def test_quota_and_global_capacity_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            try:
                first = manager.create_session("agent-a", "libero_goal", 1)
                with self.assertRaises(QuotaError):
                    manager.create_session("agent-a", "libero_goal", 1)
                with self.assertRaises(CapacityError):
                    manager.create_session("agent-b", "libero_goal", 1)
                manager.close_session(first["session_id"], "agent-a")
            finally:
                manager.shutdown()

    def test_async_create_is_idempotent_and_releases_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            try:
                first = manager.begin_session(
                    "agent-a",
                    "libero_goal",
                    1,
                    idempotency_key="request-123",
                    request_fingerprint="same-body",
                    task_id=2,
                )
                retry = manager.begin_session(
                    "agent-a",
                    "libero_goal",
                    1,
                    idempotency_key="request-123",
                    request_fingerprint="same-body",
                    task_id=2,
                )
                self.assertEqual(first["session_id"], retry["session_id"])
                record = manager._record_for_owner(first["session_id"], "agent-a")
                self.assertTrue(record.startup_done.wait(10))
                status = manager.session_status(first["session_id"], "agent-a")
                self.assertEqual(status["state"], "ready")
                self.assertEqual(status["instruction"], "mock task 2")
                manager.close_session(first["session_id"], "agent-a")
                self.assertEqual(manager.health()["slots"]["free"], 1)
            finally:
                manager.shutdown()

    def test_idempotency_key_cannot_change_request(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            try:
                created = manager.begin_session(
                    "agent-a",
                    "libero_goal",
                    1,
                    idempotency_key="request-123",
                    request_fingerprint="body-a",
                )
                with self.assertRaises(IdempotencyConflict):
                    manager.begin_session(
                        "agent-a",
                        "libero_goal",
                        1,
                        idempotency_key="request-123",
                        request_fingerprint="body-b",
                    )
                manager.close_session(created["session_id"], "agent-a")
            finally:
                manager.shutdown()

    def test_failed_sessions_do_not_consume_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            try:
                created = manager.begin_session(
                    "agent-a",
                    "libero_goal",
                    1,
                    idempotency_key="failed-001",
                    request_fingerprint="body-a",
                )
                first = manager._record_for_owner(created["session_id"], "agent-a")
                self.assertTrue(first.startup_done.wait(10))
                manager._fail_record(first, "TEST_FAILURE")
                replacement = manager.begin_session(
                    "agent-a",
                    "libero_goal",
                    1,
                    idempotency_key="replacement-001",
                    request_fingerprint="body-b",
                )
                second = manager._record_for_owner(
                    replacement["session_id"], "agent-a"
                )
                self.assertTrue(second.startup_done.wait(10))
                manager.close_session(replacement["session_id"], "agent-a")
                manager.close_session(created["session_id"], "agent-a")
            finally:
                manager.shutdown()

    def test_run_records_first_success_from_run_start(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            config = {
                "benchmark": "libero_goal",
                "task_id": 0,
                "episode_length": 5,
                "observation_level": 1,
                "bbox_scope": "initial",
            }
            try:
                run = manager.begin_run(
                    "agent-a",
                    idempotency_key="evaluation-run-001",
                    request_fingerprint="run-body",
                    config=config,
                    max_attempts=2,
                )
                time.sleep(0.02)
                created = manager.begin_session(
                    "agent-a",
                    "libero_goal",
                    1,
                    idempotency_key="episode-attempt-001",
                    request_fingerprint="session-body",
                    run_id=run["run_id"],
                    task_id=0,
                    episode_length=5,
                    observation_level=1,
                )
                record = manager._record_for_owner(
                    created["session_id"], "agent-a"
                )
                self.assertTrue(record.startup_done.wait(10))
                manager.reset(created["session_id"], "agent-a")
                for _ in range(3):
                    final = manager.step(
                        created["session_id"], "agent-a", [0] * 7
                    )
                self.assertTrue(final["terminated"])
                status = manager.run_status(run["run_id"], "agent-a")
                self.assertEqual(status["measurement_status"], "first_success_recorded")
                self.assertEqual(status["first_success_session_id"], created["session_id"])
                self.assertEqual(status["sessions_created"], 1)
                self.assertEqual(status["episodes_completed"], 1)
                self.assertEqual(status["successful_episodes"], 1)
                self.assertGreater(status["time_to_first_success_seconds"], 0.02)
                manager.close_session(created["session_id"], "agent-a")
                finished = manager.finish_run(run["run_id"], "agent-a")
                self.assertEqual(finished["state"], "finished")
            finally:
                manager.shutdown()

    def test_reset_reuses_session_and_starts_a_clean_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            config = {
                "benchmark": "libero_goal",
                "task_id": 0,
                "episode_length": 100,
                "observation_level": 1,
                "bbox_scope": "initial",
            }
            try:
                run = manager.begin_run(
                    "agent-a",
                    idempotency_key="reset-run-001",
                    request_fingerprint="run-body",
                    config=config,
                    max_attempts=2,
                )
                created = manager.begin_session(
                    "agent-a",
                    "libero_goal",
                    1,
                    idempotency_key="reset-session-001",
                    request_fingerprint="session-body",
                    run_id=run["run_id"],
                    task_id=0,
                    episode_length=100,
                    observation_level=1,
                )
                session_id = created["session_id"]
                record = manager._record_for_owner(session_id, "agent-a")
                self.assertTrue(record.startup_done.wait(10))
                first = manager.reset(session_id, "agent-a")
                self.assertEqual(first["episode_index"], 1)
                manager.step(session_id, "agent-a", [0] * 7)
                second = manager.reset(session_id, "agent-a")
                self.assertEqual(second["episode_index"], 2)
                self.assertEqual(second["observation"]["step_index"], 0)
                self.assertEqual(record.steps, 0)
                self.assertEqual(
                    manager.run_status(run["run_id"], "agent-a")["sessions_created"],
                    1,
                )
                manager.close_session(session_id, "agent-a")
                manager.finish_run(run["run_id"], "agent-a")
            finally:
                manager.shutdown()

    def test_run_locks_episode_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            config = {
                "benchmark": "libero_goal",
                "task_id": 0,
                "episode_length": 5,
                "observation_level": 1,
                "bbox_scope": "initial",
            }
            try:
                run = manager.begin_run(
                    "agent-a",
                    idempotency_key="evaluation-run-002",
                    request_fingerprint="run-body",
                    config=config,
                    max_attempts=2,
                )
                with self.assertRaises(EvaluationRunRequired):
                    manager.begin_session(
                        "agent-a",
                        "libero_goal",
                        1,
                        idempotency_key="untracked-episode",
                        request_fingerprint="untracked",
                        task_id=0,
                        episode_length=5,
                        observation_level=1,
                    )
                with self.assertRaises(RunNotFound):
                    manager.run_status(run["run_id"], "agent-b")
                with self.assertRaises(RunConfigMismatch):
                    manager.begin_session(
                        "agent-a",
                        "libero_goal",
                        1,
                        idempotency_key="episode-attempt-002",
                        request_fingerprint="different-length",
                        run_id=run["run_id"],
                        task_id=0,
                        episode_length=4,
                        observation_level=1,
                    )
            finally:
                manager.shutdown()

    def test_success_is_not_returned_when_persistence_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            config = {
                "benchmark": "libero_goal", "task_id": 0,
                "episode_length": 5, "observation_level": 1,
                "bbox_scope": "initial",
            }
            try:
                run = manager.begin_run(
                    "agent-a", idempotency_key="persistence-run",
                    request_fingerprint="run-body", config=config,
                    max_attempts=1,
                )
                created = manager.begin_session(
                    "agent-a", "libero_goal", 1,
                    idempotency_key="persistence-session",
                    request_fingerprint="session-body", run_id=run["run_id"],
                    task_id=0, episode_length=5, observation_level=1,
                )
                record = manager._record_for_owner(created["session_id"], "agent-a")
                self.assertTrue(record.startup_done.wait(10))
                manager.reset(created["session_id"], "agent-a")
                manager.step(created["session_id"], "agent-a", [0] * 7)
                manager.step(created["session_id"], "agent-a", [0] * 7)
                with mock.patch.object(
                    manager._evaluation_store,
                    "complete_attempt",
                    side_effect=OSError("database unavailable"),
                ):
                    with self.assertRaises(EvaluationPersistenceFailure):
                        manager.step(created["session_id"], "agent-a", [0] * 7)
                status = manager.run_status(run["run_id"], "agent-a")
                self.assertIsNone(status["time_to_first_success_seconds"])
                self.assertEqual(status["episodes_completed"], 0)
                manager.close_session(created["session_id"], "agent-a")
                manager.finish_run(run["run_id"], "agent-a")
            finally:
                manager.shutdown()

    def test_each_visitor_identity_runs_in_parallel(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(self._settings(directory))
            base_config = {
                "benchmark": "libero_goal",
                "task_id": 0,
                "episode_length": 5,
                "bbox_scope": "initial",
            }
            try:
                level_one = manager.begin_run(
                    "visitor-one",
                    launcher_owner="qwen",
                    idempotency_key="parallel-level-one",
                    request_fingerprint="level-one",
                    config={**base_config, "observation_level": 1},
                    max_attempts=1,
                )
                level_two = manager.begin_run(
                    "visitor-two",
                    launcher_owner="qwen",
                    idempotency_key="parallel-level-two",
                    request_fingerprint="level-two",
                    config={**base_config, "observation_level": 2},
                    max_attempts=1,
                )
                self.assertNotEqual(level_one["agent_identity"], level_two["agent_identity"])
                self.assertEqual(
                    manager.run_status(level_one["run_id"], "qwen")["state"],
                    "running",
                )
                another_level_one = manager.begin_run(
                    "visitor-three",
                    launcher_owner="qwen",
                    idempotency_key="duplicate-level-one",
                    request_fingerprint="another-level-one",
                    config={**base_config, "observation_level": 1},
                    max_attempts=1,
                )
                manager.finish_run(level_one["run_id"], "qwen")
                manager.finish_run(level_two["run_id"], "qwen")
                manager.finish_run(another_level_one["run_id"], "qwen")
            finally:
                manager.shutdown()


class EvaluationStoreTest(unittest.TestCase):
    def test_first_success_is_durable_and_compare_and_set(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "evaluations.sqlite3")
            store = EvaluationStore(path)
            config = {
                "benchmark": "libero_goal",
                "task_id": 0,
                "episode_length": 5,
                "observation_level": 1,
                "bbox_scope": "initial",
            }
            started = time.time_ns()
            run, _ = store.create_run(
                run_id="run_test_durable",
                owner="agent-a",
                idempotency_key="durable-key",
                request_fingerprint="body",
                label=None,
                config=config,
                max_attempts=2,
                started_wall_ns=started,
            )
            store.attach_session(
                run_id=run["run_id"], owner="agent-a", session_id="session-one",
                config=config, seed=1, created_wall_ns=started + 1,
            )
            store.attach_session(
                run_id=run["run_id"], owner="agent-a", session_id="session-two",
                config=config, seed=2, created_wall_ns=started + 2,
            )
            first = store.complete_attempt(
                session_id="session-one", success=True, steps=3,
                completed_wall_ns=started + 100, session_elapsed_ns=99,
                run_elapsed_ns=100,
            )
            store.mark_attempt_closed("session-one")
            second = store.complete_attempt(
                session_id="session-two", success=True, steps=3,
                completed_wall_ns=started + 200, session_elapsed_ns=198,
                run_elapsed_ns=200,
            )
            store.mark_attempt_closed("session-two")
            self.assertEqual(first["time_to_first_success_seconds"], 0.0000001)
            self.assertEqual(second["first_success_session_id"], "session-one")
            store.finish_run(
                run["run_id"], "agent-a", elapsed_ns=300,
                finished_wall_ns=started + 300,
            )
            reopened = EvaluationStore(path)
            durable = reopened.get_run(run["run_id"], "agent-a")
            self.assertEqual(durable["first_success_session_id"], "session-one")
            self.assertEqual(durable["measurement_status"], "first_success_recorded")
            self.assertEqual(durable["state"], "finished")

    def test_restart_marks_an_unfinished_run_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "evaluations.sqlite3")
            store = EvaluationStore(path)
            config = {
                "benchmark": "libero_goal", "task_id": 0,
                "episode_length": 5, "observation_level": 1,
                "bbox_scope": "initial",
            }
            store.create_run(
                run_id="run_test_interrupted", owner="agent-a",
                idempotency_key="interrupted-key", request_fingerprint="body",
                label=None, config=config, max_attempts=1,
                started_wall_ns=time.time_ns(),
            )
            reopened = EvaluationStore(path)
            status = reopened.get_run("run_test_interrupted", "agent-a")
            self.assertEqual(status["state"], "interrupted")
            self.assertEqual(status["measurement_status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
