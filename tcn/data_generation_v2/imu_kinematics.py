from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .config import REPO_ROOT


@dataclass(frozen=True)
class ImuDefinition:
    left_body: str = "femur_l"
    right_body: str = "femur_r"
    distal_local_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    world_forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    world_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    sagittal_world_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)
    left_sign: float = 1.0
    right_sign: float = 1.0
    left_standing_reference_rad: float = 0.0
    right_standing_reference_rad: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self) | {
            "angle_formula": "unwrap(atan2((xmat @ distal_axis).z, (xmat @ distal_axis).x) over episode) - standing_reference",
            "gyro_formula": "sign * world angular velocity projected on world sagittal -Y axis, rad/s",
            "angle_zero": "MuJoCo model key_qpos[0] standing femur orientation",
            "positive_direction": "validated so positive hip_flexion perturbation increases angle",
        }


def body_id(model: Any, name: str) -> int:
    if hasattr(model, "body"):
        try:
            return int(model.body(name).id)
        except Exception:
            pass
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"body not found: {name}")
    return int(bid)


def joint_qposadr(model: Any, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"joint not found: {name}")
    return int(model.jnt_qposadr[jid])


def xmat_for_body(data: Any, body_name: str) -> np.ndarray:
    try:
        return np.asarray(data.body(body_name).xmat, dtype=float).reshape(3, 3)
    except Exception:
        return np.asarray(data.xmat[body_id(data.model, body_name)], dtype=float).reshape(3, 3)


def xquat_for_body(data: Any, body_name: str) -> np.ndarray:
    try:
        return np.asarray(data.body(body_name).xquat, dtype=float).reshape(4)
    except Exception:
        return np.asarray(data.xquat[body_id(data.model, body_name)], dtype=float).reshape(4)


def raw_sagittal_angle_from_xmat(xmat: np.ndarray) -> float:
    distal = xmat @ np.asarray([1.0, 0.0, 0.0], dtype=float)
    n = np.linalg.norm(distal)
    if n > 0:
        distal = distal / n
    return float(math.atan2(float(distal[2]), float(distal[0])))


def body_world_angular_velocity(model: Any, data: Any, body_name: str) -> np.ndarray:
    out = np.zeros(6, dtype=float)
    native_model = getattr(model, "ptr", model)
    native_data = getattr(data, "ptr", data)
    mujoco.mj_objectVelocity(native_model, native_data, mujoco.mjtObj.mjOBJ_BODY, body_id(model, body_name), out, 0)
    return out[:3].copy()


def _standing_raw_angles(model_path: str | Path) -> tuple[float, float]:
    path = Path(model_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return (
        raw_sagittal_angle_from_xmat(xmat_for_body(data, "femur_l")),
        raw_sagittal_angle_from_xmat(xmat_for_body(data, "femur_r")),
    )


def _angle_delta_no_wrap(model: Any, base_qpos: np.ndarray, body: str, joint: str, delta: float) -> float:
    data0 = mujoco.MjData(model)
    data0.qpos[:] = base_qpos
    mujoco.mj_forward(model, data0)
    data1 = mujoco.MjData(model)
    data1.qpos[:] = base_qpos
    data1.qpos[joint_qposadr(model, joint)] += delta
    mujoco.mj_forward(model, data1)
    raw = np.unwrap(
        np.asarray(
            [
                raw_sagittal_angle_from_xmat(xmat_for_body(data0, body)),
                raw_sagittal_angle_from_xmat(xmat_for_body(data1, body)),
            ],
            dtype=float,
        )
    )
    return float(raw[1] - raw[0])


def build_imu_definition(model_path: str | Path) -> ImuDefinition:
    path = Path(model_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    model = mujoco.MjModel.from_xml_path(str(path))
    left_ref, right_ref = _standing_raw_angles(path)
    left_delta = _angle_delta_no_wrap(model, model.key_qpos[0].copy(), "femur_l", "hip_flexion_l", 0.1)
    right_delta = _angle_delta_no_wrap(model, model.key_qpos[0].copy(), "femur_r", "hip_flexion_r", 0.1)
    left_sign = 1.0 if left_delta >= 0.0 else -1.0
    right_sign = 1.0 if right_delta >= 0.0 else -1.0
    return ImuDefinition(
        left_sign=left_sign,
        right_sign=right_sign,
        left_standing_reference_rad=float(left_ref),
        right_standing_reference_rad=float(right_ref),
    )


def sample_raw_orientation(env: Any, imu: ImuDefinition) -> dict[str, Any]:
    model = env.sim.model
    data = env.sim.data
    left_xmat = xmat_for_body(data, imu.left_body)
    right_xmat = xmat_for_body(data, imu.right_body)
    left_w = body_world_angular_velocity(model, data, imu.left_body)
    right_w = body_world_angular_velocity(model, data, imu.right_body)
    return {
        "left_thigh_xmat": left_xmat.astype(np.float64),
        "right_thigh_xmat": right_xmat.astype(np.float64),
        "left_thigh_quat": xquat_for_body(data, imu.left_body).astype(np.float64),
        "right_thigh_quat": xquat_for_body(data, imu.right_body).astype(np.float64),
        "left_raw_sagittal_angle_rad": raw_sagittal_angle_from_xmat(left_xmat),
        "right_raw_sagittal_angle_rad": raw_sagittal_angle_from_xmat(right_xmat),
        "left_world_angular_velocity_rad_s": left_w.astype(np.float64),
        "right_world_angular_velocity_rad_s": right_w.astype(np.float64),
        "left_thigh_gyro_rad_s_raw_projected": float(imu.left_sign * -left_w[1]),
        "right_thigh_gyro_rad_s_raw_projected": float(imu.right_sign * -right_w[1]),
    }


def continuous_angles(raw_left: np.ndarray, raw_right: np.ndarray, imu: ImuDefinition) -> tuple[np.ndarray, np.ndarray]:
    left_unwrapped = np.unwrap(np.asarray(raw_left, dtype=np.float64))
    right_unwrapped = np.unwrap(np.asarray(raw_right, dtype=np.float64))
    left_ref = imu.left_standing_reference_rad + 2.0 * np.pi * np.round((left_unwrapped[0] - imu.left_standing_reference_rad) / (2.0 * np.pi))
    right_ref = imu.right_standing_reference_rad + 2.0 * np.pi * np.round((right_unwrapped[0] - imu.right_standing_reference_rad) / (2.0 * np.pi))
    return imu.left_sign * (left_unwrapped - left_ref), imu.right_sign * (right_unwrapped - right_ref)
