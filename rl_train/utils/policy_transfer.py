from __future__ import annotations

import json
import io
import zipfile
import math
from pathlib import Path

import numpy as np
import torch as th

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.policies.rl_agent_human import HumanActorCriticPolicy


ACTOR_KEYS_TO_COPY = (
    "policy_network.policy_net.0.bias",
    "policy_network.policy_net.2.weight",
    "policy_network.policy_net.2.bias",
    "policy_network.policy_net.4.weight",
    "policy_network.policy_net.4.bias",
    "log_std",
)

HUMAN_ONLY_ACTOR_LAYERS = (0, 2, 4)

HUMAN_PHASE2_TO_EXO_PHASE1_KEY_MAP = {
    f"policy_network.policy_net.{layer_index}.{suffix}":
    f"policy_network.human_policy_net.{layer_index}.{suffix}"
    for layer_index in HUMAN_ONLY_ACTOR_LAYERS
    for suffix in ("weight", "bias")
}

EXO_PHASE1_PHASE2_HUMAN_FIRST_KEY = "policy_network.human_policy_net.0.weight"
EXO_PHASE1_PHASE2_EXO_FIRST_KEY = "policy_network.exo_policy_net.0.weight"

EXO_PHASE1_PHASE2_ACTOR_DEEP_KEYS = tuple(
    f"policy_network.{actor}_policy_net.{layer_index}.{suffix}"
    for actor in ("human", "exo")
    for layer_index, suffixes in ((0, ("bias",)), (2, ("weight", "bias")), (4, ("weight", "bias")))
    for suffix in suffixes
)


def _forward_exo_phase1_actor_from_state(policy_state: dict, obs52: th.Tensor) -> th.Tensor:
    human_obs = obs52[:, :46]
    exo_obs = obs52[:, :52]
    outputs = []
    for prefix, obs in (
        ("policy_network.human_policy_net", human_obs),
        ("policy_network.exo_policy_net", exo_obs),
    ):
        x = obs
        for layer_index in (0, 2, 4):
            weight = policy_state[f"{prefix}.{layer_index}.weight"].to(obs.device)
            bias = policy_state[f"{prefix}.{layer_index}.bias"].to(obs.device)
            x = th.tanh(th.nn.functional.linear(x, weight, bias))
        outputs.append(x)
    return th.cat(outputs, dim=1)


def _load_policy_state_from_sb3_zip(checkpoint_path: Path, device: str):
    with zipfile.ZipFile(checkpoint_path) as checkpoint_zip:
        with checkpoint_zip.open("policy.pth") as policy_file:
            return th.load(io.BytesIO(policy_file.read()), map_location=device)


def _env_lpf_params(env_params) -> dict:
    enabled = bool(getattr(env_params, "exo_output_lpf_enabled", False))
    tau_s = getattr(env_params, "exo_output_lpf_tau_s", None)
    old_alpha = getattr(env_params, "exo_output_lpf_alpha", None)
    tau_s = None if tau_s is None or float(tau_s) <= 0.0 else float(tau_s)
    if tau_s is None and enabled and old_alpha is not None:
        raise ValueError(
            "exo_output_lpf_alpha is deprecated and cannot configure an enabled LPF. "
            "Use exo_output_lpf_tau_s."
        )
    if tau_s is None and enabled:
        raise ValueError("exo_output_lpf_tau_s is required when exo_output_lpf_enabled is true")
    tau_s = 0.1 if tau_s is None else float(tau_s)
    control_hz = float(getattr(env_params, "control_framerate"))
    dt_s = 1.0 / control_hz
    return {
        "exo_output_lpf_enabled": enabled,
        "exo_output_lpf_tau_s": tau_s,
        "exo_output_lpf_dt_s": dt_s,
        "exo_output_lpf_alpha_effective": 1.0 - math.exp(-dt_s / tau_s),
    }


def _find_session_config_for_checkpoint(checkpoint_path: Path) -> Path | None:
    for parent in [checkpoint_path.parent, *checkpoint_path.parents]:
        candidate = parent / "session_config.json"
        if candidate.exists():
            return candidate
    return None


def _read_lpf_params_from_session_config(checkpoint_path: Path) -> dict | None:
    session_config_path = _find_session_config_for_checkpoint(checkpoint_path)
    if session_config_path is None:
        return None
    with session_config_path.open("r", encoding="utf-8") as f:
        env_params = json.load(f).get("env_params", {})
    enabled = bool(env_params.get("exo_output_lpf_enabled", False))
    tau_s = env_params.get("exo_output_lpf_tau_s")
    tau_s = None if tau_s is None or float(tau_s) <= 0.0 else float(tau_s)
    if tau_s is None and enabled and "exo_output_lpf_alpha" in env_params:
        raise ValueError(
            f"{session_config_path} uses deprecated exo_output_lpf_alpha for an enabled LPF. "
            "Use exo_output_lpf_tau_s."
        )
    if tau_s is None and enabled:
        raise ValueError(f"{session_config_path} must set exo_output_lpf_tau_s when LPF is enabled")
    tau_s = 0.1 if tau_s is None else float(tau_s)
    dt_s = 1.0 / float(env_params["control_framerate"])
    return {
        "exo_output_lpf_enabled": enabled,
        "exo_output_lpf_tau_s": tau_s,
        "exo_output_lpf_dt_s": dt_s,
        "exo_output_lpf_alpha_effective": 1.0 - math.exp(-dt_s / tau_s),
        "source_session_config": str(session_config_path),
    }


def _lpf_params_match(left: dict | None, right: dict | None) -> bool:
    if left is None or right is None:
        return True
    left_enabled = bool(left.get("exo_output_lpf_enabled", False))
    right_enabled = bool(right.get("exo_output_lpf_enabled", False))
    if left_enabled != right_enabled:
        return False
    if not left_enabled and not right_enabled:
        return True
    return (
        abs(float(left.get("exo_output_lpf_tau_s", 0.1)) - float(right.get("exo_output_lpf_tau_s", 0.1)))
        <= 1e-12
        and abs(float(left.get("exo_output_lpf_dt_s", 0.0)) - float(right.get("exo_output_lpf_dt_s", 0.0)))
        <= 1e-12
    )


def _forward_human_phase1_actor_from_state(policy_state: dict, obs44: th.Tensor) -> th.Tensor:
    x = obs44
    for layer_index in (0, 2, 4):
        weight = policy_state[f"policy_network.policy_net.{layer_index}.weight"].to(obs44.device)
        bias = policy_state[f"policy_network.policy_net.{layer_index}.bias"].to(obs44.device)
        x = th.tanh(th.nn.functional.linear(x, weight, bias))
    return x


def _forward_human_only_actor_from_state(policy_state: dict, obs: th.Tensor) -> th.Tensor:
    x = obs
    for layer_index in HUMAN_ONLY_ACTOR_LAYERS:
        weight = policy_state[f"policy_network.policy_net.{layer_index}.weight"].to(obs.device)
        bias = policy_state[f"policy_network.policy_net.{layer_index}.bias"].to(obs.device)
        x = th.tanh(th.nn.functional.linear(x, weight, bias))
    return x


def transfer_human_phase1_to_phase2_checkpoint(
    *,
    source_checkpoint: str,
    target_config: str,
    output_checkpoint: str,
    report_path: str,
) -> dict:
    source_path = Path(source_checkpoint)
    output_path = Path(output_checkpoint)
    if not source_path.exists():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("output checkpoint must not overwrite source checkpoint")

    with open(target_config, "r") as config_file:
        env_id = json.load(config_file)["env_params"]["env_id"]
    config_type = EnvironmentHandler.get_config_type_from_session_id(env_id)
    config = EnvironmentHandler.get_session_config_from_path(target_config, config_type)
    config.env_params.prev_trained_policy_path = None
    config.env_params.num_envs = 1
    config.ppo_params.device = "cpu"
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    try:
        new_model = EnvironmentHandler.get_stable_baselines3_model(config, env)

        old_state = _load_policy_state_from_sb3_zip(source_path, config.ppo_params.device)
        new_state = new_model.policy.state_dict()

        old_first = old_state["policy_network.policy_net.0.weight"]
        new_first = new_state["policy_network.policy_net.0.weight"]
        if tuple(old_first.shape) != (new_first.shape[0], 44):
            raise ValueError(f"unexpected old first-layer shape: {tuple(old_first.shape)}")
        if tuple(new_first.shape) != (old_first.shape[0], 46):
            raise ValueError(f"unexpected new first-layer shape: {tuple(new_first.shape)}")

        transferred_state = {key: value.clone() for key, value in new_state.items()}
        transferred_state["policy_network.policy_net.0.weight"][:, :44] = old_first
        transferred_state["policy_network.policy_net.0.weight"][:, 44:46] = 0

        copied_keys = ["policy_network.policy_net.0.weight"]
        for key in ACTOR_KEYS_TO_COPY:
            if key not in old_state or key not in transferred_state:
                raise KeyError(f"missing parameter key for transfer: {key}")
            if tuple(old_state[key].shape) != tuple(transferred_state[key].shape):
                raise ValueError(
                    f"shape mismatch for {key}: old={tuple(old_state[key].shape)} "
                    f"new={tuple(transferred_state[key].shape)}"
                )
            transferred_state[key] = old_state[key].clone()
            copied_keys.append(key)

        new_model.policy.load_state_dict(transferred_state, strict=True)

        obs44 = th.linspace(-0.5, 0.5, 44, dtype=old_first.dtype, device=old_first.device).reshape(1, 44)
        obs46 = th.cat(
            [obs44.to(new_first.device), th.zeros(1, 2, dtype=new_first.dtype, device=new_first.device)],
            dim=1,
        )
        with th.no_grad():
            old_mean = _forward_human_phase1_actor_from_state(old_state, obs44).detach().cpu()
            new_mean = new_model.policy.policy_network.forward_actor(obs46).detach().cpu()
        max_abs_difference = float(th.max(th.abs(old_mean - new_mean)).item())
        if not np.isfinite(max_abs_difference) or max_abs_difference >= 1e-6:
            raise ValueError(f"actor consistency check failed: {max_abs_difference}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_model.save(str(output_path))

        report = {
            "source_checkpoint": str(source_path),
            "output_checkpoint": str(output_path),
            "target_config": target_config,
            "old_first_layer_shape": list(old_first.shape),
            "new_first_layer_shape": list(new_first.shape),
            "new_columns_zero": bool(
                th.max(th.abs(transferred_state["policy_network.policy_net.0.weight"][:, 44:46])).item() == 0
            ),
            "copied_keys": copied_keys,
            "not_copied": [
                "policy_network.value_net.*",
                "policy.optimizer",
                "rollout_buffer",
                "learning_rate_scheduler_state",
            ],
            "actor_consistency_max_abs_difference": max_abs_difference,
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report
    finally:
        try:
            env.close()
        except Exception:
            pass


def transfer_human_phase2_to_exo_phase1_checkpoint(
    *,
    source_checkpoint: str,
    target_config: str,
    output_checkpoint: str,
    report_path: str,
) -> dict:
    source_path = Path(source_checkpoint)
    output_path = Path(output_checkpoint)
    if not source_path.exists():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("output checkpoint must not overwrite source checkpoint")

    with open(target_config, "r") as config_file:
        env_id = json.load(config_file)["env_params"]["env_id"]
    config_type = EnvironmentHandler.get_config_type_from_session_id(env_id)
    config = EnvironmentHandler.get_session_config_from_path(target_config, config_type)
    config.env_params.prev_trained_policy_path = None
    config.env_params.num_envs = 1
    config.ppo_params.device = "cpu"
    config.ppo_params.n_steps = 8
    config.ppo_params.batch_size = 8
    transfer_lpf_tau_s = getattr(config.env_params, "exo_output_lpf_tau_s", None)
    if (
        not bool(getattr(config.env_params, "exo_output_lpf_enabled", False))
        and (transfer_lpf_tau_s is None or float(transfer_lpf_tau_s) <= 0.0)
    ):
        config.env_params.exo_output_lpf_tau_s = 0.1

    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    try:
        target_model = EnvironmentHandler.get_stable_baselines3_model(config, env)

        source_state = _load_policy_state_from_sb3_zip(source_path, "cpu")
        target_state = target_model.policy.state_dict()
        transferred_state = {key: value.clone() for key, value in target_state.items()}

        copied_keys = []
        for source_key, target_key in HUMAN_PHASE2_TO_EXO_PHASE1_KEY_MAP.items():
            if source_key not in source_state:
                raise KeyError(f"missing source parameter: {source_key}")
            if target_key not in transferred_state:
                raise KeyError(f"missing target parameter: {target_key}")
            if tuple(source_state[source_key].shape) != tuple(transferred_state[target_key].shape):
                raise ValueError(
                    f"shape mismatch for {source_key} -> {target_key}: "
                    f"source={tuple(source_state[source_key].shape)} "
                    f"target={tuple(transferred_state[target_key].shape)}"
                )
            transferred_state[target_key] = source_state[source_key].clone()
            copied_keys.append((source_key, target_key))

        source_log_std = source_state["log_std"]
        target_log_std = transferred_state["log_std"].clone()
        if tuple(source_log_std.shape) != (22,):
            raise ValueError(f"expected Human Phase 2 log_std shape (22,), got {tuple(source_log_std.shape)}")
        if tuple(target_log_std.shape) != (24,):
            raise ValueError(f"expected Exo Phase 1 log_std shape (24,), got {tuple(target_log_std.shape)}")
        target_log_std[:22] = source_log_std
        transferred_state["log_std"] = target_log_std

        target_model.policy.load_state_dict(transferred_state, strict=True)
        EnvironmentHandler.apply_exo_teacher_freeze_if_requested(config, target_model)

        obs46 = th.linspace(-0.5, 0.5, 46, dtype=source_state["policy_network.policy_net.0.weight"].dtype).reshape(1, 46)
        obs52 = th.cat([obs46, th.zeros(1, 6, dtype=obs46.dtype)], dim=1)
        with th.no_grad():
            source_mean = _forward_human_only_actor_from_state(source_state, obs46).detach().cpu()
            target_human_obs = target_model.policy.policy_network.network_index_handler.map_observation_to_network(
                obs52.to(target_model.policy.device),
                "human_actor",
            )
            target_mean = target_model.policy.policy_network.human_policy_net(target_human_obs).detach().cpu()
        human_actor_max_abs_difference = float(th.max(th.abs(source_mean - target_mean)).item())
        log_std_max_abs_difference = float(
            th.max(th.abs(source_log_std.detach().cpu() - target_log_std[:22].detach().cpu())).item()
        )
        if not np.isfinite(human_actor_max_abs_difference) or human_actor_max_abs_difference >= 1e-6:
            raise ValueError(f"human actor consistency check failed: {human_actor_max_abs_difference}")
        if not np.isfinite(log_std_max_abs_difference) or log_std_max_abs_difference >= 1e-6:
            raise ValueError(f"human log_std consistency check failed: {log_std_max_abs_difference}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        target_model.save(str(output_path))

        report = {
            "source_checkpoint": str(source_path),
            "output_checkpoint": str(output_path),
            "target_config": target_config,
            "copied_actor_keys": copied_keys,
            "copied_log_std_slice": "log_std[0:22]",
            "new_log_std_slice": "log_std[22:24]",
            "not_copied": [
                "source policy_network.value_net.*",
                "source optimizer state",
                "source rollout buffer",
                "source scheduler state",
                "target policy_network.exo_policy_net.*",
                "target policy_network.value_net.*",
            ],
            "human_actor_consistency_max_abs_difference": human_actor_max_abs_difference,
            "human_log_std_consistency_max_abs_difference": log_std_max_abs_difference,
            "target_log_std_shape": list(target_log_std.shape),
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report
    finally:
        try:
            env.close()
        except Exception:
            pass


def transfer_exo_phase1_to_phase2_checkpoint(
    *,
    source_checkpoint: str,
    target_config: str,
    output_checkpoint: str,
    report_path: str,
) -> dict:
    source_path = Path(source_checkpoint)
    output_path = Path(output_checkpoint)
    if not source_path.exists():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("output checkpoint must not overwrite source checkpoint")

    with open(target_config, "r") as config_file:
        env_id = json.load(config_file)["env_params"]["env_id"]
    config_type = EnvironmentHandler.get_config_type_from_session_id(env_id)
    config = EnvironmentHandler.get_session_config_from_path(target_config, config_type)
    source_lpf_params = _read_lpf_params_from_session_config(source_path)
    target_lpf_params = _env_lpf_params(config.env_params)
    lpf_params_match = _lpf_params_match(source_lpf_params, target_lpf_params)
    if source_lpf_params is not None and not lpf_params_match:
        raise ValueError(
            "Exo Phase 1 -> Exo Phase 2 LPF parameter mismatch: "
            f"source={source_lpf_params}, target={target_lpf_params}"
        )
    config.env_params.prev_trained_policy_path = None
    config.env_params.num_envs = 1
    config.ppo_params.device = "cpu"
    config.ppo_params.n_steps = 8
    config.ppo_params.batch_size = 8

    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    try:
        target_model = EnvironmentHandler.get_stable_baselines3_model(config, env)

        source_state = _load_policy_state_from_sb3_zip(source_path, "cpu")
        target_state = target_model.policy.state_dict()
        transferred_state = {key: value.clone() for key, value in target_state.items()}

        source_human_first = source_state[EXO_PHASE1_PHASE2_HUMAN_FIRST_KEY]
        target_human_first = transferred_state[EXO_PHASE1_PHASE2_HUMAN_FIRST_KEY]
        source_exo_first = source_state[EXO_PHASE1_PHASE2_EXO_FIRST_KEY]
        target_exo_first = transferred_state[EXO_PHASE1_PHASE2_EXO_FIRST_KEY]
        if tuple(source_human_first.shape) != (target_human_first.shape[0], 46):
            raise ValueError(f"unexpected Exo Phase 1 human first-layer shape: {tuple(source_human_first.shape)}")
        if tuple(target_human_first.shape) != (source_human_first.shape[0], 48):
            raise ValueError(f"unexpected Exo Phase 2 human first-layer shape: {tuple(target_human_first.shape)}")
        if tuple(source_exo_first.shape) != (target_exo_first.shape[0], 52):
            raise ValueError(f"unexpected Exo Phase 1 exo first-layer shape: {tuple(source_exo_first.shape)}")
        if tuple(target_exo_first.shape) != tuple(source_exo_first.shape):
            raise ValueError(f"unexpected Exo Phase 2 exo first-layer shape: {tuple(target_exo_first.shape)}")

        transferred_state[EXO_PHASE1_PHASE2_HUMAN_FIRST_KEY][:, :46] = source_human_first
        transferred_state[EXO_PHASE1_PHASE2_HUMAN_FIRST_KEY][:, 46:48] = 0
        transferred_state[EXO_PHASE1_PHASE2_EXO_FIRST_KEY] = source_exo_first.clone()

        copied_keys = [
            EXO_PHASE1_PHASE2_HUMAN_FIRST_KEY,
            EXO_PHASE1_PHASE2_EXO_FIRST_KEY,
        ]
        for key in EXO_PHASE1_PHASE2_ACTOR_DEEP_KEYS:
            if key not in source_state or key not in transferred_state:
                raise KeyError(f"missing parameter key for Exo Phase 1 -> Exo Phase 2 transfer: {key}")
            if tuple(source_state[key].shape) != tuple(transferred_state[key].shape):
                raise ValueError(
                    f"shape mismatch for {key}: source={tuple(source_state[key].shape)} "
                    f"target={tuple(transferred_state[key].shape)}"
                )
            transferred_state[key] = source_state[key].clone()
            copied_keys.append(key)

        if tuple(source_state["log_std"].shape) != tuple(transferred_state["log_std"].shape):
            raise ValueError(
                f"log_std shape mismatch: source={tuple(source_state['log_std'].shape)} "
                f"target={tuple(transferred_state['log_std'].shape)}"
            )
        transferred_state["log_std"] = source_state["log_std"].clone()
        copied_keys.append("log_std")
        log_std_max_abs_difference = float(
            th.max(th.abs(transferred_state["log_std"].detach().cpu() - source_state["log_std"].detach().cpu())).item()
        )

        target_model.policy.load_state_dict(transferred_state, strict=True)

        obs52 = th.linspace(-0.5, 0.5, 52, dtype=source_human_first.dtype).reshape(1, 52)
        obs54 = th.cat(
            [
                obs52,
                th.tensor([[0.125, -0.25]], dtype=obs52.dtype),
            ],
            dim=1,
        )
        with th.no_grad():
            source_mean = _forward_exo_phase1_actor_from_state(source_state, obs52).detach().cpu()
            target_mean = target_model.policy.policy_network.forward_actor(
                obs54.to(target_model.policy.device)
            ).detach().cpu()

        human_max_abs_difference = float(th.max(th.abs(source_mean[:, :22] - target_mean[:, :22])).item())
        exo_max_abs_difference = float(th.max(th.abs(source_mean[:, 22:24] - target_mean[:, 22:24])).item())
        human_new_columns_max_abs = float(
            th.max(th.abs(transferred_state[EXO_PHASE1_PHASE2_HUMAN_FIRST_KEY][:, 46:48])).item()
        )
        exo_direct_copy = bool(
            th.equal(
                transferred_state[EXO_PHASE1_PHASE2_EXO_FIRST_KEY].detach().cpu(),
                source_exo_first.detach().cpu(),
            )
        )

        value_key = "policy_network.value_net.0.weight"
        critic_fresh = bool(
            value_key in source_state
            and value_key in target_state
            and not th.equal(transferred_state[value_key].detach().cpu(), source_state[value_key].detach().cpu())
        )
        optimizer_fresh = True
        human_unfrozen = bool(
            all(param.requires_grad for param in target_model.policy.policy_network.human_policy_net.parameters())
            and target_model.policy.log_std.requires_grad
        )

        if human_max_abs_difference >= 1e-6:
            raise ValueError(f"Exo Phase 1 -> Exo Phase 2 human equivalence failed: {human_max_abs_difference}")
        if exo_max_abs_difference >= 1e-6:
            raise ValueError(f"Exo Phase 1 -> Exo Phase 2 exo equivalence failed: {exo_max_abs_difference}")
        if human_new_columns_max_abs != 0:
            raise ValueError("Exo Phase 2 transfer new human first-layer columns are not zero")
        if not exo_direct_copy:
            raise ValueError("Exo Phase 2 exo first layer was not copied directly")
        if log_std_max_abs_difference >= 1e-6:
            raise ValueError(f"Exo Phase 2 log_std inheritance failed: {log_std_max_abs_difference}")
        if not critic_fresh:
            raise ValueError("Exo Phase 2 critic freshness check failed")
        if not human_unfrozen:
            raise ValueError("Exo Phase 2 human actor/log_std should be trainable")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        target_model.save(str(output_path))

        report = {
            "source_checkpoint": str(source_path),
            "output_checkpoint": str(output_path),
            "target_config": target_config,
            "source_lpf_params": source_lpf_params,
            "target_lpf_params": target_lpf_params,
            "lpf_params_match": lpf_params_match,
            "source_human_first_layer_shape": list(source_human_first.shape),
            "target_human_first_layer_shape": list(target_human_first.shape),
            "source_exo_first_layer_shape": list(source_exo_first.shape),
            "target_exo_first_layer_shape": list(target_exo_first.shape),
            "copied_actor_keys": copied_keys,
            "copied_log_std": True,
            "human_equivalence_max_abs_diff": human_max_abs_difference,
            "exo_equivalence_max_abs_diff": exo_max_abs_difference,
            "human_new_columns_max_abs": human_new_columns_max_abs,
            "human_new_column_indices": [46, 47],
            "exo_direct_copy": exo_direct_copy,
            "log_std_max_abs_diff": log_std_max_abs_difference,
            "critic_fresh": critic_fresh,
            "optimizer_fresh": optimizer_fresh,
            "rollout_buffer_inherited": False,
            "human_unfrozen": human_unfrozen,
            "not_copied": [
                "policy_network.value_net.*",
                "policy.optimizer state from source",
                "rollout_buffer",
                "learning_rate_scheduler_state",
            ],
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report
    finally:
        try:
            env.close()
        except Exception:
            pass
