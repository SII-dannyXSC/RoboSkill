from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .observations import public_observation
from .phases import PhaseCondition, PhaseTracker, phases_for_task
from .settings import Settings


BENCHMARK_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


@lru_cache(maxsize=None)
def task_catalog(backend: str, benchmark: str) -> Tuple[Dict[str, Any], ...]:
    """Return public task metadata without exposing BDDL paths or init states."""
    if benchmark not in BENCHMARK_TASK_COUNTS:
        raise ValueError("unsupported benchmark")
    if backend == "mock":
        return tuple(
            {"task_id": task_id, "instruction": f"mock task {task_id}"}
            for task_id in range(BENCHMARK_TASK_COUNTS[benchmark])
        )
    from libero.libero import benchmark as benchmark_module

    benchmark_class = benchmark_module.get_benchmark_dict()[benchmark]
    suite = benchmark_class()
    return tuple(
        {
            "task_id": task_id,
            "instruction": suite.get_task(task_id).language,
        }
        for task_id in range(BENCHMARK_TASK_COUNTS[benchmark])
    )


@dataclass(frozen=True)
class EpisodeOptions:
    episode_length: int
    observation_level: int = 1
    bbox_scope: str = "initial"


class EpisodeBackend(ABC):
    @property
    @abstractmethod
    def instruction(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Tuple[float, ...]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def result(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class InvalidSessionState(Exception):
    pass


def validate_action(action: Any) -> Tuple[float, ...]:
    if not isinstance(action, (list, tuple)) or len(action) != 7:
        raise ValueError("action must contain exactly 7 numbers")
    values = tuple(float(value) for value in action)
    if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values):
        raise ValueError("each action value must be finite and within [-1, 1]")
    return values


class MockEpisode(EpisodeBackend):
    def __init__(
        self,
        settings: Settings,
        benchmark: str,
        task_id: int,
        seed: int,
        gpu_id: int,
        options: Optional[EpisodeOptions] = None,
    ):
        self.settings = settings
        self.benchmark = benchmark
        self.task_id = task_id
        self.seed = seed
        self.gpu_id = gpu_id
        self.options = options or EpisodeOptions(settings.default_episode_steps)
        self.steps = 0
        self.started = False
        self.ended = False
        self.success = False
        self.closed = False
        self._phase_tracker = PhaseTracker(phases_for_task(benchmark, task_id))

    @property
    def instruction(self) -> str:
        return f"mock task {self.task_id}"

    def _raw_obs(self) -> Dict[str, Any]:
        value = self.steps % 255
        image = np.full(
            (self.settings.image_height, self.settings.image_width, 3),
            value,
            dtype=np.uint8,
        )
        return {
            "agentview_image": image,
            "robot0_eye_in_hand_image": image,
            "robot0_joint_pos": np.zeros(7, dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
            "robot0_joint_vel": np.zeros(7, dtype=np.float32),
            "robot0_gripper_qvel": np.zeros(2, dtype=np.float32),
            "robot0_joint_commanded_torque": np.zeros(7, dtype=np.float32),
            "robot0_eef_force": np.zeros(3, dtype=np.float32),
            "robot0_eef_torque": np.zeros(3, dtype=np.float32),
            "robot0_eef_velocity": np.zeros(6, dtype=np.float32),
            # Deliberate privileged fields exercise the serializer allowlist.
            "object_positions": {"secret_object": [1.0, 2.0, 3.0]},
            "mujoco_state": np.arange(10),
        }

    def _public(self, *, initial: bool) -> Dict[str, Any]:
        level = self.options.observation_level
        include_bboxes = level >= 2 and (
            initial or self.options.bbox_scope == "every_frame"
        )
        height, width = self.settings.image_height, self.settings.image_width
        bboxes = None
        if include_bboxes:
            box = {
                "label": "mock_target_1",
                "xyxy": [1, 1, max(2, width // 2), max(2, height // 2)],
                "normalized_xyxy": [0.0, 0.0, 0.5, 0.5],
                "visible_pixels": max(1, width * height // 4),
            }
            bboxes = {"agentview_rgb": [box], "wrist_rgb": [box]}
        depth = None
        calibration = None
        if level >= 4:
            depth = {
                "agentview_depth": np.ones((height, width), dtype=np.float32),
                "wrist_depth": np.ones((height, width), dtype=np.float32),
            }
            calibration = {
                name: {
                    "intrinsic": np.eye(3).tolist(),
                    "camera_to_world": np.eye(4).tolist(),
                    "convention": "opencv",
                }
                for name in ("agentview_rgb", "wrist_rgb")
            }
        return public_observation(
            self._raw_obs(),
            quality=self.settings.jpeg_quality,
            frame_id=self.steps,
            step_index=self.steps,
            level=level,
            depth_maps=depth,
            camera_calibration=calibration,
            object_bboxes=bboxes,
        )

    def reset(self) -> Dict[str, Any]:
        self.started = True
        self.steps = 0
        self.ended = False
        self.success = False
        self._phase_tracker.reset()
        return {
            "observation": self._public(initial=True),
            "terminated": False,
            "truncated": False,
            "_phase_successes": list(self._phase_tracker.statuses()),
        }

    def step(self, action: Tuple[float, ...]) -> Dict[str, Any]:
        if not self.started:
            raise InvalidSessionState("session must be reset before step")
        if self.ended:
            raise InvalidSessionState("episode has already ended")
        validate_action(action)
        self.steps += 1
        self.success = self.steps >= 3
        self._phase_tracker.update(
            self.steps,
            lambda phase: self.steps >= min(phase.phase_id, 3),
        )
        success_monotonic_ns = time.monotonic_ns() if self.success else None
        truncated = self.steps >= self.options.episode_length and not self.success
        self.ended = self.success or truncated
        return {
            "observation": self._public(initial=False),
            "terminated": self.success,
            "truncated": truncated,
            "_phase_successes": list(self._phase_tracker.statuses()),
            "_success_monotonic_ns": success_monotonic_ns,
        }

    def result(self) -> Dict[str, Any]:
        status = "completed" if self.ended else ("running" if self.started else "created")
        return {
            "status": status,
            "success": self.success if self.ended else None,
            "steps": self.steps,
            "_phase_successes": list(self._phase_tracker.statuses()),
        }

    def close(self) -> None:
        self.closed = True


class LiberoEpisode(EpisodeBackend):
    CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
    PUBLIC_CAMERA_NAMES = ("agentview_rgb", "wrist_rgb")

    def __init__(
        self,
        settings: Settings,
        benchmark: str,
        task_id: int,
        seed: int,
        gpu_id: int,
        options: EpisodeOptions,
    ):
        # Imports happen only inside a child process, after process/GPU setup.
        from libero.libero import benchmark as benchmark_module
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_class = benchmark_module.get_benchmark_dict()[benchmark]
        self._benchmark = benchmark_class()
        self._task = self._benchmark.get_task(task_id)
        self._init_states = self._benchmark.get_task_init_states(task_id)
        self._init_index = seed % len(self._init_states)
        self._settings = settings
        self._options = options
        self._steps = 0
        self._started = False
        self._success = False
        self._ended = False
        self._closed = False
        self._phase_tracker = PhaseTracker(phases_for_task(benchmark, task_id))
        self._initial_object_heights: Dict[str, float] = {}
        self._latest_eef_position = np.zeros(3, dtype=float)

        env_args = {
            "bddl_file_name": self._benchmark.get_task_bddl_file_path(task_id),
            "camera_heights": settings.image_height,
            "camera_widths": settings.image_width,
            "camera_names": list(self.CAMERA_NAMES),
            "camera_depths": options.observation_level >= 4,
            # Segmentation is used internally only to derive task-object bboxes.
            # Bboxes use an internal on-demand segmentation render. Enabling
            # robosuite camera_segmentations mutates site sizes in 1.4.0.
            "camera_segmentations": None,
            "use_object_obs": False,
            "render_gpu_device_id": gpu_id,
            "horizon": options.episode_length,
            "ignore_done": False,
        }
        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(seed)
        self._objects_of_interest = tuple(self._env.obj_of_interest)

    def _object_position(self, object_name: str) -> np.ndarray:
        obj = self._env.env.get_object(object_name)
        body_id = self._env.sim.model.body_name2id(obj.root_body)
        return np.asarray(self._env.sim.data.body_xpos[body_id], dtype=float)

    def _phase_condition_satisfied(self, condition: PhaseCondition) -> bool:
        if condition.kind == "success":
            return self._success
        if condition.kind == "predicate":
            return bool(self._env.env._eval_predicate(condition.arguments))
        object_name = condition.arguments[0]
        if condition.kind == "grasped":
            obj = self._env.env.get_object(object_name)
            return bool(
                self._env.env._check_grasp(
                    self._env.env.robots[0].gripper,
                    obj.contact_geoms,
                )
            )
        object_position = self._object_position(object_name)
        if condition.kind == "lifted":
            initial_height = self._initial_object_heights[object_name]
            return object_position[2] >= initial_height + 0.02
        if condition.kind == "near":
            return bool(
                np.linalg.norm(self._latest_eef_position - object_position) <= 0.08
            )
        raise ValueError(f"unsupported phase condition: {condition.kind}")

    def _update_phase_successes(self) -> None:
        self._phase_tracker.update(
            self._steps,
            lambda phase: self._phase_condition_satisfied(phase.condition),
        )

    @property
    def instruction(self) -> str:
        return self._task.language

    def _extended_proprio(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if self._options.observation_level < 3:
            return raw
        enriched = dict(raw)
        robot = self._env.robots[0]
        extra = {
            # This is the controller-applied command, not a physical sensor.
            "robot0_joint_commanded_torque": robot.torques,
            "robot0_eef_force": robot.ee_force,
            "robot0_eef_torque": robot.ee_torque,
            "robot0_eef_velocity": robot._hand_total_velocity,
        }
        for key, value in extra.items():
            if value is not None:
                enriched[key] = np.asarray(value)
        return enriched

    def _metric_depth(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        from robosuite.utils.camera_utils import get_real_depth_map

        result: Dict[str, Any] = {}
        for public_name, camera_name in zip(
            ("agentview_depth", "wrist_depth"), self.CAMERA_NAMES
        ):
            raw_name = f"{camera_name}_depth"
            if raw_name not in raw:
                raise RuntimeError(f"missing depth observation: {raw_name}")
            result[public_name] = get_real_depth_map(self._env.sim, raw[raw_name])
        return result

    def _camera_calibration(self) -> Dict[str, Any]:
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
        )

        result: Dict[str, Any] = {}
        for public_name, camera_name in zip(
            self.PUBLIC_CAMERA_NAMES, self.CAMERA_NAMES
        ):
            intrinsic = get_camera_intrinsic_matrix(
                self._env.sim,
                camera_name,
                self._settings.image_height,
                self._settings.image_width,
            )
            camera_to_world = get_camera_extrinsic_matrix(
                self._env.sim, camera_name
            )
            result[public_name] = {
                "intrinsic": np.asarray(intrinsic, dtype=float).tolist(),
                "camera_to_world": np.asarray(camera_to_world, dtype=float).tolist(),
                "convention": "opencv",
            }
        return result

    def _object_bboxes(self, _raw: Dict[str, Any]) -> Dict[str, Any]:
        height, width = self._settings.image_height, self._settings.image_width
        result: Dict[str, Any] = {}
        for public_name, camera_name in zip(
            self.PUBLIC_CAMERA_NAMES, self.CAMERA_NAMES
        ):
            segmentation = self._env.sim.render(
                camera_name=camera_name,
                width=width,
                height=height,
                depth=False,
                segmentation=True,
            )
            # Channel 1 contains geom IDs. Match the public RGB orientation.
            geom_ids_image = np.asarray(segmentation)[::-1, :, 1]
            boxes = []
            for label in self._objects_of_interest:
                instance = self._env.env.model.instances_to_ids.get(label, {})
                geom_ids = instance.get("geom", ())
                if not geom_ids:
                    continue
                ys, xs = np.where(np.isin(geom_ids_image, geom_ids))
                if not len(xs):
                    continue
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
                boxes.append(
                    {
                        "label": label,
                        "xyxy": [x1, y1, x2, y2],
                        "normalized_xyxy": [
                            x1 / width,
                            y1 / height,
                            x2 / width,
                            y2 / height,
                        ],
                        "visible_pixels": int(len(xs)),
                    }
                )
            result[public_name] = boxes
        return result

    def _public(self, raw: Dict[str, Any], *, initial: bool) -> Dict[str, Any]:
        level = self._options.observation_level
        include_bboxes = level >= 2 and (
            initial or self._options.bbox_scope == "every_frame"
        )
        return public_observation(
            self._extended_proprio(raw),
            quality=self._settings.jpeg_quality,
            frame_id=self._steps,
            step_index=self._steps,
            level=level,
            depth_maps=self._metric_depth(raw) if level >= 4 else None,
            camera_calibration=self._camera_calibration() if level >= 4 else None,
            object_bboxes=self._object_bboxes(raw) if include_bboxes else None,
        )

    def reset(self) -> Dict[str, Any]:
        self._env.reset()
        raw = self._env.set_init_state(self._init_states[self._init_index])
        self._steps = 0
        self._started = True
        self._ended = False
        self._success = bool(self._env.check_success())
        self._latest_eef_position = np.asarray(raw["robot0_eef_pos"], dtype=float)
        self._initial_object_heights = {
            condition.arguments[0]: self._object_position(condition.arguments[0])[2]
            for phase in self._phase_tracker.phases
            for condition in (phase.condition,)
            if condition.kind == "lifted"
        }
        self._phase_tracker.reset()
        self._update_phase_successes()
        success_monotonic_ns = time.monotonic_ns() if self._success else None
        self._ended = self._success
        return {
            "observation": self._public(raw, initial=True),
            "terminated": self._success,
            "truncated": False,
            "_phase_successes": list(self._phase_tracker.statuses()),
            "_success_monotonic_ns": success_monotonic_ns,
        }

    def step(self, action: Tuple[float, ...]) -> Dict[str, Any]:
        if not self._started:
            raise InvalidSessionState("session must be reset before step")
        if self._ended:
            raise InvalidSessionState("episode has already ended")
        values = validate_action(action)
        raw, _reward, _done, _info = self._env.step(
            np.asarray(values, dtype=np.float32)
        )
        self._steps += 1
        self._success = bool(self._env.check_success())
        self._latest_eef_position = np.asarray(raw["robot0_eef_pos"], dtype=float)
        self._update_phase_successes()
        success_monotonic_ns = time.monotonic_ns() if self._success else None
        truncated = (
            self._steps >= self._options.episode_length and not self._success
        )
        # robosuite's `done` at this horizon is a time-limit signal. Keep the
        # Gymnasium distinction: task success terminates; the step limit truncates.
        self._ended = bool(self._success or truncated)
        return {
            "observation": self._public(raw, initial=False),
            "terminated": self._success,
            "truncated": truncated,
            "_phase_successes": list(self._phase_tracker.statuses()),
            "_success_monotonic_ns": success_monotonic_ns,
        }

    def result(self) -> Dict[str, Any]:
        return {
            "status": (
                "completed"
                if self._ended
                else ("running" if self._started else "created")
            ),
            "success": self._success if self._ended else None,
            "steps": self._steps,
            "_phase_successes": list(self._phase_tracker.statuses()),
        }

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True


def create_episode(
    settings: Settings,
    benchmark: str,
    task_id: int,
    seed: int,
    gpu_id: int,
    options: EpisodeOptions,
) -> EpisodeBackend:
    if settings.backend == "mock":
        return MockEpisode(settings, benchmark, task_id, seed, gpu_id, options)
    if settings.backend == "libero":
        return LiberoEpisode(settings, benchmark, task_id, seed, gpu_id, options)
    raise ValueError("unsupported backend")
