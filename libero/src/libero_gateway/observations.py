from __future__ import annotations

import base64
import io
import zlib
from typing import Any, Dict, Iterable, Optional

import numpy as np
from PIL import Image


PUBLIC_IMAGE_KEYS = {
    "agentview_rgb": "agentview_image",
    "wrist_rgb": "robot0_eye_in_hand_image",
}
PUBLIC_DEPTH_KEYS = {
    "agentview_depth": "agentview_depth",
    "wrist_depth": "robot0_eye_in_hand_depth",
}
BASE_PROPRIO_KEYS = (
    "robot0_joint_pos",
    "robot0_gripper_qpos",
    "robot0_eef_pos",
    "robot0_eef_quat",
)
EXTENDED_PROPRIO_KEYS = (
    "robot0_joint_vel",
    "robot0_gripper_qvel",
    "robot0_joint_commanded_torque",
    "robot0_eef_force",
    "robot0_eef_torque",
    "robot0_eef_velocity",
)


class ObservationError(Exception):
    pass


def encode_jpeg(array: Any, quality: int, flip_vertical: bool = True) -> str:
    image = np.asarray(array)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ObservationError("public RGB observation has invalid shape")
    if flip_vertical:
        image = image[::-1]
    image = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality, optimize=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_float32_array(array: Any, *, flip_vertical: bool = False) -> Dict[str, Any]:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ObservationError("depth observation has invalid values")
    if flip_vertical:
        values = values[::-1]
    values = np.ascontiguousarray(values.astype("<f4", copy=False))
    return {
        "encoding": "float32-zlib-base64",
        "dtype": "float32",
        "shape": list(values.shape),
        "unit": "meter",
        "base64": base64.b64encode(zlib.compress(values.tobytes())).decode("ascii"),
    }


def _finite_vector(
    raw: Dict[str, Any], key: str, *, allow_unavailable: bool = False
) -> Optional[list]:
    if key not in raw:
        if allow_unavailable:
            return None
        raise ObservationError(f"required proprioception is unavailable: {key}")
    values = np.asarray(raw[key], dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(values)):
        # Some physical sensors (notably force/torque) are undefined before
        # the first control step. Omit them rather than inventing zeros.
        if allow_unavailable:
            return None
        raise ObservationError("non-finite proprioception")
    return values.tolist()


def public_observation(
    raw: Dict[str, Any],
    *,
    quality: int,
    frame_id: int,
    step_index: int,
    level: int = 1,
    depth_maps: Optional[Dict[str, Any]] = None,
    camera_calibration: Optional[Dict[str, Any]] = None,
    object_bboxes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize only explicitly allowlisted fields for the selected level."""
    images: Dict[str, Any] = {}
    for public_name, raw_name in PUBLIC_IMAGE_KEYS.items():
        if raw_name not in raw:
            raise ObservationError(f"required public observation is unavailable: {public_name}")
        images[public_name] = {
            "media_type": "image/jpeg",
            "base64": encode_jpeg(raw[raw_name], quality),
        }

    proprioception: Dict[str, Any] = {}
    keys: Iterable[str] = BASE_PROPRIO_KEYS
    if level >= 3:
        keys = BASE_PROPRIO_KEYS + EXTENDED_PROPRIO_KEYS
    for key in keys:
        values = _finite_vector(
            raw, key, allow_unavailable=key in EXTENDED_PROPRIO_KEYS
        )
        if values is not None:
            proprioception[key] = values

    result: Dict[str, Any] = {
        "images": images,
        "proprioception": proprioception,
        "frame_id": frame_id,
        "step_index": step_index,
    }
    if level >= 2 and object_bboxes is not None:
        result["annotations"] = {"object_bboxes": object_bboxes}
    if level >= 4:
        if depth_maps is None or camera_calibration is None:
            raise ObservationError("level 4 requires depth and camera calibration")
        result["depth"] = {
            name: encode_float32_array(values, flip_vertical=True)
            for name, values in depth_maps.items()
        }
        result["camera_calibration"] = camera_calibration
    return result
