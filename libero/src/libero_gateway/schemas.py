from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


BenchmarkName = Literal[
    "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"
]
ObservationLevel = Literal[1, 2, 3, 4]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSessionRequest(StrictModel):
    """Legacy v1 request; uses server defaults and the optional v1 fixed task."""

    benchmark: BenchmarkName


class TaskSelection(StrictModel):
    benchmark: BenchmarkName
    task_id: Optional[int] = Field(default=None, ge=0)


class ObservationRequest(StrictModel):
    level: ObservationLevel = 1
    bbox_scope: Literal["initial", "every_frame"] = "initial"


class CreateSessionV2Request(StrictModel):
    run_id: str = Field(min_length=8, max_length=64)
    task: TaskSelection
    episode_length: Optional[int] = Field(default=None, ge=1)
    observation: ObservationRequest = Field(default_factory=ObservationRequest)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)


class ActionSpec(StrictModel):
    shape: List[int] = Field(default_factory=lambda: [7])
    low: List[float] = Field(default_factory=lambda: [-1.0] * 7)
    high: List[float] = Field(default_factory=lambda: [1.0] * 7)
    order: List[str] = Field(
        default_factory=lambda: [
            "dx",
            "dy",
            "dz",
            "rotation_vector_x",
            "rotation_vector_y",
            "rotation_vector_z",
            "gripper",
        ]
    )
    frame: Literal["mujoco_world"] = "mujoco_world"
    controller: Literal["OSC_POSE_delta"] = "OSC_POSE_delta"
    control_hz: int = 20
    translation_scale_meters: float = 0.05
    rotation_scale_radians: float = 0.5
    gripper_negative: Literal["open"] = "open"
    gripper_positive: Literal["close"] = "close"


class ActionSpecV1(StrictModel):
    shape: List[int] = Field(default_factory=lambda: [7])
    low: List[float] = Field(default_factory=lambda: [-1.0] * 7)
    high: List[float] = Field(default_factory=lambda: [1.0] * 7)


class ObservationSpecV1(StrictModel):
    cameras: Dict[str, List[int]]
    proprioception_keys: List[str]
    image_encoding: Literal["jpeg"] = "jpeg"
    image_orientation: Literal["upright"] = "upright"


class CreateSessionV1Response(StrictModel):
    session_id: str
    instruction: str
    action_spec: ActionSpecV1
    observation_spec: ObservationSpecV1


class ObservationSpec(StrictModel):
    level: ObservationLevel = 1
    cameras: Dict[str, List[int]]
    required_proprioception_keys: List[str]
    optional_proprioception_keys: List[str] = Field(default_factory=list)
    image_encoding: Literal["jpeg"] = "jpeg"
    image_orientation: Literal["upright"] = "upright"
    bbox_scope: Optional[Literal["initial", "every_frame"]] = None
    depth_encoding: Optional[Literal["float32-zlib-base64"]] = None
    camera_calibration: bool = False


class ImagePayload(StrictModel):
    media_type: Literal["image/jpeg"] = "image/jpeg"
    base64: str


class ArrayPayload(StrictModel):
    encoding: Literal["float32-zlib-base64"] = "float32-zlib-base64"
    dtype: Literal["float32"] = "float32"
    shape: List[int]
    unit: Literal["meter"] = "meter"
    base64: str


class CameraCalibration(StrictModel):
    intrinsic: List[List[float]]
    camera_to_world: List[List[float]]
    convention: Literal["opencv"] = "opencv"


class BoundingBox(StrictModel):
    label: str
    xyxy: List[int]
    normalized_xyxy: List[float]
    visible_pixels: int


class ObservationAnnotations(StrictModel):
    object_bboxes: Dict[str, List[BoundingBox]]


class PublicObservation(StrictModel):
    images: Dict[str, ImagePayload]
    proprioception: Dict[str, List[float]]
    frame_id: int
    step_index: int
    depth: Optional[Dict[str, ArrayPayload]] = None
    camera_calibration: Optional[Dict[str, CameraCalibration]] = None
    annotations: Optional[ObservationAnnotations] = None


class SessionConfig(StrictModel):
    run_id: Optional[str] = None
    benchmark: BenchmarkName
    task_id: int
    episode_length: int
    observation_level: ObservationLevel
    bbox_scope: Optional[Literal["initial", "every_frame"]] = None
    seed: Optional[int] = None


class SessionTiming(StrictModel):
    created_at: str
    ready_at: Optional[str] = None
    startup_seconds: Optional[float] = None
    completed_at: Optional[str] = None
    time_to_success_seconds: Optional[float] = None


class CreateSessionResponse(StrictModel):
    session_id: str
    state: Literal[
        "starting", "ready", "running", "completed", "failed", "closing", "cancelled"
    ]
    instruction: Optional[str] = None
    error_code: Optional[str] = None
    action_spec: ActionSpec
    observation_spec: ObservationSpec
    session_config: SessionConfig
    timing: SessionTiming


class SessionStatusResponse(CreateSessionResponse):
    pass


class ResetResponse(StrictModel):
    observation: PublicObservation
    terminated: bool = False
    truncated: bool = False
    episode_index: int = Field(default=1, ge=1)


class StepRequest(StrictModel):
    action: List[
        Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
    ] = Field(min_length=7, max_length=7)
    expected_step_index: Optional[int] = Field(default=None, ge=0)


class StepResponse(StrictModel):
    observation: PublicObservation
    terminated: bool
    truncated: bool


class ResultResponse(StrictModel):
    status: Literal["created", "running", "completed", "closed", "failed"]
    success: Optional[bool] = None
    steps: int


class TaskInfo(StrictModel):
    task_id: int
    instruction: str


class TaskCatalogResponse(StrictModel):
    benchmark: BenchmarkName
    tasks: List[TaskInfo]


class HarnessPhaseProgress(StrictModel):
    phase_id: int = Field(ge=1)
    name: str
    description: str
    success: bool
    first_achieved_step: Optional[int] = Field(default=None, ge=0)


class HarnessTaskProgressResponse(StrictModel):
    benchmark: BenchmarkName
    task_id: int
    completed_phases: int = Field(ge=0)
    total_phases: int = Field(ge=0)
    completion_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    phases: List[HarnessPhaseProgress]


class StartRunRequest(StrictModel):
    label: Optional[str] = Field(default=None, min_length=1, max_length=128)
    task: TaskSelection
    episode_length: Optional[int] = Field(default=None, ge=1)
    observation: ObservationRequest = Field(default_factory=ObservationRequest)
    max_attempts: int = Field(default=100, ge=1, le=1000)


class AllocateExperimentRequest(StrictModel):
    """Trusted local allocation request; no pre-existing API identity required."""

    label: Optional[str] = Field(default=None, min_length=1, max_length=128)
    task: TaskSelection
    episode_length: Optional[int] = Field(default=None, ge=1)
    observation: ObservationRequest = Field(default_factory=ObservationRequest)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)


class EvaluationRunConfig(StrictModel):
    benchmark: BenchmarkName
    task_id: int
    episode_length: int
    observation_level: ObservationLevel
    bbox_scope: Literal["initial", "every_frame"]


class EvaluationRunResponse(StrictModel):
    run_id: str
    agent_identity: str
    agent_token: Optional[str] = None
    state: Literal["running", "finished", "interrupted"]
    measurement_status: Literal[
        "measuring",
        "first_success_recorded",
        "finished_without_success",
        "interrupted",
    ]
    label: Optional[str] = None
    config: EvaluationRunConfig
    max_attempts: int
    started_at: str
    finished_at: Optional[str] = None
    elapsed_seconds: float
    first_success_at: Optional[str] = None
    time_to_first_success_seconds: Optional[float] = None
    first_success_session_id: Optional[str] = None
    sessions_created: int
    episodes_completed: int
    successful_episodes: int
    active_sessions: int
    clock_source: Literal["server_monotonic"]


class AllocateExperimentResponse(StrictModel):
    run: EvaluationRunResponse
    session: CreateSessionResponse
    agent_token: str


class EpisodeLengthCapability(StrictModel):
    default: int
    maximum: int


class ObservationLevelCapability(StrictModel):
    minimum: int
    maximum: int
    profiles: Dict[str, str]


class CapabilitiesResponse(StrictModel):
    worker_mode: Literal["per_session"]
    credential_type: Literal["static", "run", "session"]
    scopes: List[str]
    benchmarks: Dict[str, Dict[str, int]]
    observation_levels: ObservationLevelCapability
    episode_length: EpisodeLengthCapability
    max_concurrent_sessions: int
    fixed_seed_allowed: bool
    timing_metrics: List[str]


class ErrorBody(StrictModel):
    code: str
    request_id: str


class ErrorResponse(StrictModel):
    error: ErrorBody
