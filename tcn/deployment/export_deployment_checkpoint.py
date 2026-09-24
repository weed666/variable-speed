from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch


DEFAULT_OUTPUT = Path(__file__).resolve().with_name("unified_tcn_latest_100hz_deploy.pt")
SUPPORTED_MODES = {"steady", "acceleration", "deceleration", "unified"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_count(state_dict: dict[str, torch.Tensor]) -> int:
    return int(sum(v.numel() for v in state_dict.values()))


def build_package(source_path: Path) -> dict[str, Any]:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    required = [
        "model_type",
        "mode",
        "state_dict",
        "model_config",
        "sensor_hz",
        "history_steps",
        "input_channel_names",
        "input_mean",
        "input_std",
        "normalization_scheme",
        "torque_scale_nm",
    ]
    missing = [k for k in required if k not in source]
    if missing:
        raise KeyError(f"source checkpoint missing required keys: {missing}")

    state_dict = source["state_dict"]
    config = dict(source["model_config"])
    history_steps = int(source["history_steps"])
    sensor_hz = int(source["sensor_hz"])
    torque_scale = float(source["torque_scale_nm"])
    rf = int(config.get("receptive_field_samples", 1 + 2 * (int(config["kernel_size"]) - 1) * sum(config["dilations"])))

    if source["model_type"] != "causal_tcn":
        raise ValueError(f"expected causal_tcn source, got {source['model_type']}")
    mode = str(source["mode"])
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"expected one of {sorted(SUPPORTED_MODES)}, got {source['mode']}")
    if history_steps != 100 or sensor_hz != 100:
        raise ValueError(f"expected 100 Hz / 100 history, got {sensor_hz} / {history_steps}")

    return {
        "deployment_format_version": 1,
        "model_type": "causal_tcn",
        "mode": mode,
        "state_dict": state_dict,
        "model_config": config,
        "sensor_hz": sensor_hz,
        "control_hz": 100,
        "history_steps": history_steps,
        "history_duration_s": history_steps / float(sensor_hz),
        "input_channel_names": list(source["input_channel_names"]),
        "input_order": list(source["input_channel_names"]),
        "input_mean": list(map(float, source["input_mean"])),
        "input_std": list(map(float, source["input_std"])),
        "normalization_scheme": str(source["normalization_scheme"]),
        "output_names": ["left_exo_action_norm", "right_exo_action_norm"],
        "output_activation": "tanh",
        "output_range": [-1.0, 1.0],
        "torque_scale_nm": torque_scale,
        "input_units": {
            "angle": "rad",
            "angular_velocity": "rad/s",
        },
        "input_semantics": {
            "forward_thigh_flexion_positive": True,
            "standing_relative_angle": True,
            "left_channel_sign_default": "controller CLI configurable",
            "right_channel_sign_default": "controller CLI configurable",
        },
        "checkpoint_provenance": {
            "source_best_model_path": str(source_path),
            "source_mode": mode,
            "best_epoch": int(source.get("best_epoch", source.get("epoch", -1))),
            "best_val_mse": float(source.get("best_val_mse", float("nan"))),
            "source_checkpoint_sha256": sha256_file(source_path),
        },
        "architecture": {
            "input_channels": int(config["input_channels"]),
            "hidden_channels": int(config["hidden_channels"]),
            "output_channels": int(config["output_channels"]),
            "kernel_size": int(config["kernel_size"]),
            "dilations": list(map(int, config["dilations"])),
            "dropout": float(config.get("dropout", 0.0)),
            "activation": str(config.get("activation", "silu")),
            "receptive_field_samples": rf,
            "parameter_count": tensor_count(state_dict),
        },
        "safety": {
            "action_min": -1.0,
            "action_max": 1.0,
            "nominal_torque_scale_nm": torque_scale,
            "command_min_nm": -torque_scale,
            "command_max_nm": torque_scale,
            "slew_limiter_enabled": False,
            "max_delta_nm_per_step": None,
            "note": "Raw 100 Hz TCN closed-loop baseline is preserved; validate any future slew limiter separately.",
        },
    }


def verify_state_dict_identical(source_path: Path, output_path: Path) -> float:
    src = torch.load(source_path, map_location="cpu", weights_only=False)["state_dict"]
    dep = torch.load(output_path, map_location="cpu", weights_only=False)["state_dict"]
    if src.keys() != dep.keys():
        raise AssertionError("state_dict keys differ")
    max_diff = 0.0
    for key in src:
        if src[key].shape != dep[key].shape or src[key].dtype != dep[key].dtype:
            raise AssertionError(f"state_dict metadata differs for {key}")
        diff = (src[key] - dep[key]).abs().max().item() if src[key].numel() else 0.0
        max_diff = max(max_diff, float(diff))
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Unified causal TCN deployment checkpoint.")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source_checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    package = build_package(source)
    torch.save(package, output)
    max_diff = verify_state_dict_identical(source, output)
    print(f"deployment_pt={output}")
    print(f"source_checkpoint={source}")
    print(f"state_dict_max_abs_difference={max_diff:.12g}")
    print(f"parameter_count={package['architecture']['parameter_count']}")


if __name__ == "__main__":
    main()
