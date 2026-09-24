from __future__ import annotations

import copy
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from stable_baselines3 import PPO

from .config import PHYSICS_HZ, REPO_ROOT, TEACHER_HZ, PipelineConfig
from .imu_kinematics import ImuDefinition, build_imu_definition, continuous_angles, sample_raw_orientation

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rl_train.envs  # noqa: F401,E402
from myosuite.utils import gym  # noqa: E402
from rl_train.envs.environment_handler import EnvironmentHandler  # noqa: E402
from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy  # noqa: E402
from rl_train.utils.data_types import DictionableDataclass  # noqa: E402


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    velocity_m_s: float
    seed: int
    reference_index: int
    duration_s: float
    initial_state_perturbation: dict[str, Any] | None = None


def teacher_result_dir(checkpoint: Path) -> Path:
    return checkpoint.resolve().parents[1]


def load_teacher_config(checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    result_dir = teacher_result_dir(checkpoint)
    with (result_dir / "session_config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_class = EnvironmentHandler.get_config_type_from_session_id(config_dict["env_params"]["env_id"])
    return DictionableDataclass.create(config_class, config_dict), config_dict


def make_runtime_config(base_config: Any, *, seed: int, duration_s: float) -> Any:
    cfg = copy.deepcopy(base_config)
    cfg.env_params.num_envs = 1
    cfg.env_params.seed = int(seed)
    cfg.env_params.control_framerate = TEACHER_HZ
    cfg.env_params.physics_sim_framerate = PHYSICS_HZ
    cfg.env_params.episode_duration_s = float(duration_s)
    cfg.env_params.custom_max_episode_steps = int(round(float(duration_s) * TEACHER_HZ))
    cfg.env_params.ramp_start_time_min = 0.05
    cfg.env_params.ramp_start_time_max = max(0.05, min(0.1, float(duration_s) * 0.1))
    cfg.env_params.min_ramp_duration = min(
        float(getattr(cfg.env_params, "min_ramp_duration", 1.0)),
        max(0.05, float(duration_s) * 0.2),
    )
    cfg.env_params.max_ramp_duration = max(
        cfg.env_params.min_ramp_duration,
        max(0.1, float(duration_s) - cfg.env_params.ramp_start_time_max - 0.1),
    )
    cfg.ppo_params.device = "cpu"
    cfg.ppo_params.n_steps = max(64, min(2048, cfg.env_params.custom_max_episode_steps))
    cfg.ppo_params.batch_size = cfg.ppo_params.n_steps
    return cfg


def make_env(config: Any) -> tuple[Any, Any]:
    ref_data = EnvironmentHandler.load_reference_data(config)
    env = gym.make(
        config.env_params.env_id,
        seed=config.env_params.seed,
        model_path=config.env_params.model_path,
        env_params=config.env_params,
        reference_data=ref_data,
        is_evaluate_mode=True,
    ).unwrapped
    return env, ref_data


def load_model(checkpoint: Path, env: Any) -> Any:
    model = PPO.load(
        str(checkpoint),
        env=env,
        custom_objects={"policy_class": HumanExoActorCriticPolicy},
        device="cpu",
    )
    EnvironmentHandler.restore_sb3_save_params(model)
    return model


def set_fixed_steady_task(env: Any, spec: EpisodeSpec) -> None:
    if hasattr(env, "set_fixed_evaluation_task"):
        env.set_fixed_evaluation_task(
            initial_velocity=spec.velocity_m_s,
            goal_velocity=spec.velocity_m_s,
            signed_acceleration=0.0,
            ramp_start_time=spec.duration_s + 1.0,
            reference_index=spec.reference_index,
        )
    elif hasattr(env, "set_target_velocity_mode_manually"):
        env.set_target_velocity_mode_manually(
            env.VelocityMode.CONSTANT,
            starting_phase=0.0,
            initial_target_velocity=spec.velocity_m_s,
            min_target_velocity=spec.velocity_m_s,
            max_target_velocity=spec.velocity_m_s,
        )
    else:
        raise RuntimeError("environment does not expose a fixed steady task API")


def apply_initial_state_perturbation(env: Any, perturbation: dict[str, Any] | None) -> None:
    if not perturbation:
        return
    qpos_scale = float(perturbation.get("qpos_scale", 0.0))
    qvel_scale = float(perturbation.get("qvel_scale", 0.0))
    seed = int(perturbation.get("seed", 0))
    rng = np.random.default_rng(seed)
    qpos_delta = rng.normal(0.0, qpos_scale, size=env.sim.data.qpos.shape)
    qvel_delta = rng.normal(0.0, qvel_scale, size=env.sim.data.qvel.shape)
    for joint_name in ("pelvis_tx", "pelvis_ty"):
        try:
            qpos_delta[env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(joint_name)]] = 0.0
        except Exception:
            pass
    try:
        qvel_delta[env.sim.model.jnt_dofadr[env.sim.model.joint_name2id("pelvis_tx")]] = 0.0
    except Exception:
        pass
    env.sim.data.qpos[:] = env.sim.data.qpos[:] + qpos_delta
    env.sim.data.qvel[:] = env.sim.data.qvel[:] + qvel_delta
    env.sim.forward()
    if hasattr(env, "robot") and hasattr(env.robot, "sync_sims"):
        env.robot.sync_sims(env.sim, env.sim_obsd)


def current_observation(env: Any) -> Any:
    if hasattr(env, "get_obs"):
        return env.get_obs()
    if hasattr(env, "get_obs_dict") and hasattr(env, "obsdict2obsvec"):
        obs_dict = env.get_obs_dict(env.sim)
        return env.obsdict2obsvec(obs_dict, env.obs_keys)[1]
    raise RuntimeError("environment does not expose current observation API")


def stage_info(env: Any) -> dict[str, Any]:
    return env.get_stage3_diagnostics() if hasattr(env, "get_stage3_diagnostics") else {}


def exo_info(env: Any) -> dict[str, Any]:
    return env.get_exo_teacher_diagnostics() if hasattr(env, "get_exo_teacher_diagnostics") else {}


def safe_joint(env: Any, name: str, kind: str) -> float:
    try:
        j = env.sim.data.joint(name)
        return float(j.qpos[0] if kind == "qpos" else j.qvel[0])
    except Exception:
        return float("nan")


def sensor_contact(env: Any, *names: str) -> bool:
    for name in names:
        try:
            if float(env.sim.data.sensor(name).data.copy()[0]) > 0.0:
                return True
        except Exception:
            pass
    return False


def action_label(env: Any, action: np.ndarray, config: Any) -> tuple[float, float, str]:
    action = np.asarray(action, dtype=float).reshape(-1)
    raw_right = float(action[22]) if action.size >= 24 else float("nan")
    raw_left = float(action[23]) if action.size >= 24 else float("nan")
    enabled = bool(getattr(config.env_params, "exo_output_lpf_enabled", False))
    if not enabled:
        return raw_left, raw_right, "raw_actor_action_indices_23_left_22_right"
    exo = exo_info(env)
    executed = exo.get("latest_normalized_exo_action")
    if executed is None or len(executed) < 2:
        raise RuntimeError("LPF enabled but latest_normalized_exo_action unavailable")
    return float(executed[1]), float(executed[0]), "executed_exo_diagnostics"


def make_specs(cfg: PipelineConfig) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    per_v = cfg.effective_episodes_per_velocity()
    rng = random.Random(cfg.seed)
    for v_idx, velocity in enumerate(cfg.velocities()):
        for ep_idx in range(per_v):
            seed = int(cfg.seed + v_idx * 100_000 + ep_idx)
            specs.append(
                EpisodeSpec(
                    episode_id=f"steady_v{int(round(velocity * 100)):03d}_ep{ep_idx:04d}",
                    velocity_m_s=float(velocity),
                    seed=seed,
                    reference_index=rng.randrange(0, 90),
                    duration_s=float(cfg.episode_duration),
                )
            )
    return specs


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float("nan")
    except Exception:
        return float("nan")


def run_episode(base_config: Any, checkpoint: Path, spec: EpisodeSpec, imu: ImuDefinition) -> dict[str, Any]:
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    env, _ = make_env(make_runtime_config(base_config, seed=spec.seed, duration_s=spec.duration_s))
    model = load_model(checkpoint, env)
    rows: list[dict[str, Any]] = []
    termination = {"reason": "full_horizon", "terminated": False, "truncated": False, "fall": False}
    try:
        set_fixed_steady_task(env, spec)
        obs, _ = env.reset(seed=spec.seed)
        apply_initial_state_perturbation(env, spec.initial_state_perturbation)
        if spec.initial_state_perturbation:
            obs = current_observation(env)
        expected_steps = int(round(spec.duration_s * TEACHER_HZ))
        for update_idx in range(expected_steps):
            t_before = float(env.sim.data.time)
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=float).reshape(-1)
            obs, _reward, terminated, truncated, info = env.step(action)
            stage = stage_info(env)
            exo = exo_info(env)
            left_action, right_action, action_source = action_label(env, action, base_config)
            orient = sample_raw_orientation(env, imu)
            torques = exo.get("exo_joint_torque", [float("nan"), float("nan")])
            rows.append(
                {
                    "time_s": float(env.sim.data.time),
                    "teacher_update_time_s": t_before,
                    "teacher_update_index": update_idx,
                    "goal_velocity_m_s": spec.velocity_m_s,
                    "actual_pelvis_velocity_m_s": _finite_or_nan(
                        stage.get("actual_pelvis_velocity", safe_joint(env, "pelvis_tx", "qvel"))
                    ),
                    "target_velocity_m_s": _finite_or_nan(stage.get("target_velocity", spec.velocity_m_s)),
                    "task_phase": str(stage.get("phase", "steady")),
                    "left_exo_action": left_action,
                    "right_exo_action": right_action,
                    "action_source": action_source,
                    "left_exo_torque_nm": float(torques[1]) if len(torques) > 1 else float("nan"),
                    "right_exo_torque_nm": float(torques[0]) if len(torques) > 0 else float("nan"),
                    "left_foot_contact": sensor_contact(env, "l_foot", "l_toes"),
                    "right_foot_contact": sensor_contact(env, "r_foot", "r_toes"),
                    "left_hip_angle_rad": safe_joint(env, "hip_flexion_l", "qpos"),
                    "right_hip_angle_rad": safe_joint(env, "hip_flexion_r", "qpos"),
                    **orient,
                }
            )
            if terminated or truncated:
                termination = {
                    "reason": str(stage.get("termination_reason") or info.get("termination_reason") or ("terminated" if terminated else "truncated")),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "fall": bool(terminated and safe_joint(env, "pelvis_ty", "qpos") < getattr(env, "_safe_height", -np.inf)),
                }
                break
        return {"spec": asdict(spec), "rows": rows, "termination": termination}
    finally:
        try:
            model.env = None
            env.close()
        except Exception:
            pass


def write_raw_h5(path: Path, cfg: PipelineConfig, base_config_dict: dict[str, Any], imu: ImuDefinition, episodes: list[dict[str, Any]]) -> None:
    if path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "tcn_data_generation_v2_raw_30hz_steady"
        h5.attrs["teacher_frequency_hz"] = TEACHER_HZ
        h5.attrs["physics_frequency_hz"] = PHYSICS_HZ
        h5.attrs["teacher_checkpoint"] = str(cfg.checkpoint)
        h5.attrs["teacher_result_dir"] = str(teacher_result_dir(cfg.checkpoint))
        h5.attrs["environment_config_json"] = json.dumps(base_config_dict, sort_keys=True)
        h5.attrs["imu_definition_json"] = json.dumps(imu.to_json(), sort_keys=True)
        h5.attrs["smoke_test"] = bool(cfg.smoke_test)
        root = h5.create_group("episodes")
        for i, payload in enumerate(episodes):
            rows = payload["rows"]
            spec = payload["spec"]
            grp = root.create_group(f"{i:06d}")
            grp.attrs["episode_id"] = spec["episode_id"]
            grp.attrs["random_seed"] = int(spec["seed"])
            grp.attrs["goal_velocity_m_s"] = float(spec["velocity_m_s"])
            grp.attrs["episode_duration_s"] = float(spec["duration_s"])
            grp.attrs["reference_index"] = int(spec["reference_index"])
            grp.attrs["regime"] = "steady"
            grp.attrs["initial_velocity_m_s"] = float(spec["velocity_m_s"])
            grp.attrs["target_acceleration_m_s2"] = 0.0
            grp.attrs["signed_acceleration_m_s2"] = 0.0
            grp.attrs["delta_velocity_m_s"] = 0.0
            grp.attrs["task_id"] = "steady"
            grp.attrs["initial_state_perturbation_json"] = json.dumps(spec.get("initial_state_perturbation"), sort_keys=True)
            grp.attrs["termination_reason"] = payload["termination"]["reason"]
            grp.attrs["termination_json"] = json.dumps(payload["termination"], sort_keys=True)
            if not rows:
                continue
            raw_left = np.asarray([r["left_raw_sagittal_angle_rad"] for r in rows], dtype=np.float64)
            raw_right = np.asarray([r["right_raw_sagittal_angle_rad"] for r in rows], dtype=np.float64)
            left_angle, right_angle = continuous_angles(raw_left, raw_right, imu)
            field_values: dict[str, Any] = {
                "time_s": np.asarray([r["time_s"] for r in rows], dtype=np.float64),
                "teacher_update_time_s": np.asarray([r["teacher_update_time_s"] for r in rows], dtype=np.float64),
                "teacher_update_index": np.asarray([r["teacher_update_index"] for r in rows], dtype=np.int32),
                "goal_velocity_m_s": np.asarray([r["goal_velocity_m_s"] for r in rows], dtype=np.float32),
                "actual_pelvis_velocity_m_s": np.asarray([r["actual_pelvis_velocity_m_s"] for r in rows], dtype=np.float32),
                "target_velocity_m_s": np.asarray([r["target_velocity_m_s"] for r in rows], dtype=np.float32),
                "left_thigh_xmat": np.stack([r["left_thigh_xmat"] for r in rows]).astype(np.float64),
                "right_thigh_xmat": np.stack([r["right_thigh_xmat"] for r in rows]).astype(np.float64),
                "left_thigh_quat": np.stack([r["left_thigh_quat"] for r in rows]).astype(np.float64),
                "right_thigh_quat": np.stack([r["right_thigh_quat"] for r in rows]).astype(np.float64),
                "left_raw_sagittal_angle_rad": raw_left,
                "right_raw_sagittal_angle_rad": raw_right,
                "left_thigh_angle_rad": left_angle.astype(np.float32),
                "right_thigh_angle_rad": right_angle.astype(np.float32),
                "left_thigh_gyro_rad_s": np.asarray([r["left_thigh_gyro_rad_s_raw_projected"] for r in rows], dtype=np.float32),
                "right_thigh_gyro_rad_s": np.asarray([r["right_thigh_gyro_rad_s_raw_projected"] for r in rows], dtype=np.float32),
                "left_world_angular_velocity_rad_s": np.stack([r["left_world_angular_velocity_rad_s"] for r in rows]).astype(np.float32),
                "right_world_angular_velocity_rad_s": np.stack([r["right_world_angular_velocity_rad_s"] for r in rows]).astype(np.float32),
                "left_exo_action": np.asarray([r["left_exo_action"] for r in rows], dtype=np.float32),
                "right_exo_action": np.asarray([r["right_exo_action"] for r in rows], dtype=np.float32),
                "left_exo_torque_nm": np.asarray([r["left_exo_torque_nm"] for r in rows], dtype=np.float32),
                "right_exo_torque_nm": np.asarray([r["right_exo_torque_nm"] for r in rows], dtype=np.float32),
                "left_foot_contact": np.asarray([r["left_foot_contact"] for r in rows], dtype=np.bool_),
                "right_foot_contact": np.asarray([r["right_foot_contact"] for r in rows], dtype=np.bool_),
                "left_hip_angle_rad": np.asarray([r["left_hip_angle_rad"] for r in rows], dtype=np.float32),
                "right_hip_angle_rad": np.asarray([r["right_hip_angle_rad"] for r in rows], dtype=np.float32),
            }
            for key, value in field_values.items():
                grp.create_dataset(key, data=value)
            phase = np.asarray([str(r["task_phase"]).encode("utf-8") for r in rows])
            source = np.asarray([str(r["action_source"]).encode("utf-8") for r in rows])
            grp.create_dataset("gait_phase", data=phase)
            grp.create_dataset("action_source", data=source)


def collect(cfg: PipelineConfig) -> dict[str, Any]:
    if not cfg.checkpoint.exists():
        raise FileNotFoundError(cfg.checkpoint)
    base_config, base_config_dict = load_teacher_config(cfg.checkpoint)
    imu = build_imu_definition(base_config.env_params.model_path)
    specs = make_specs(cfg)
    episodes = []
    for idx, spec in enumerate(specs, 1):
        print(f"[collect30] {idx}/{len(specs)} {spec.episode_id} v={spec.velocity_m_s:.2f} seed={spec.seed}", flush=True)
        episodes.append(run_episode(base_config, cfg.checkpoint, spec, imu))
    write_raw_h5(cfg.raw_path, cfg, base_config_dict, imu, episodes)
    return {
        "raw_path": str(cfg.raw_path),
        "episodes": len(episodes),
        "velocities": cfg.velocities(),
        "episodes_per_velocity": cfg.effective_episodes_per_velocity(),
        "imu_definition": imu.to_json(),
    }


def main(argv: list[str] | None = None) -> int:
    from .config import parse_args

    cfg = parse_args(argv)
    print(collect(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
