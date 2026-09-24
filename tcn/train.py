#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcn.data import create_dataloader  # noqa: E402
from tcn.models import CausalTCN  # noqa: E402
from tcn.training import Trainer, set_seed  # noqa: E402

MODES = ("steady", "acceleration", "deceleration", "unified")
CONFIG_PATH = REPO_ROOT / "tcn" / "configs" / "train.json"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_model(config: dict[str, Any]) -> CausalTCN:
    return CausalTCN(
        input_channels=int(config["input_channels"]),
        hidden_channels=int(config["hidden_channels"]),
        output_channels=int(config["output_channels"]),
        kernel_size=int(config["kernel_size"]),
        dilations=list(config["dilations"]),
        dropout=float(config["dropout"]),
        activation=str(config["activation"]),
    )


def build_loaders(config: dict[str, Any]):
    common = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": bool(config.get("pin_memory", True)),
        "persistent_workers": bool(config.get("persistent_workers", True)),
        "normalize": bool(config.get("normalize", True)),
        "history_steps": int(config["history_steps"]),
        "unified_normalization": str(config.get("unified_normalization", "regime_balanced")),
        "dataset_root": config.get("dataset_root"),
        "normalization_path": config.get("normalization_path"),
    }
    train_loader = create_dataloader(
        mode=config["mode"],
        split="train",
        unified_balance=bool(config.get("unified_balance", True)),
        **common,
    )
    val_loader = create_dataloader(
        mode=config["mode"],
        split="val",
        unified_balance=False,
        **common,
    )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified causal TCN trainer")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable batch-level progress output")
    parser.add_argument("--smoke-test", action="store_true", help="Run a few train/val batches and exit")
    parser.add_argument("--smoke-batches", type=int, default=2)
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(config)
    mapping = {
        "mode": "mode",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "learning_rate": "learning_rate",
        "max_epochs": "max_epochs",
        "seed": "seed",
        "device": "device",
    }
    for arg_name, cfg_name in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            out[cfg_name] = value
    if getattr(args, "no_progress", False):
        out["show_progress"] = False
    if out["mode"] not in MODES:
        raise ValueError(f"Unsupported mode: {out['mode']}")
    return out


def output_dir_for(config: dict[str, Any], run_name: str | None) -> Path:
    if run_name is None:
        run_name = time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / config.get("results_dir", "tcn/results") / config["mode"] / run_name


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["seed"]), deterministic=bool(config.get("deterministic", False)))
    device = resolve_device(str(config.get("device", "auto")))
    model = build_model(config)
    train_loader, val_loader = build_loaders(config)

    if args.smoke_test:
        output_dir = REPO_ROOT / "tcn" / "results" / "_smoke_test" / config["mode"]
    else:
        output_dir = output_dir_for(config, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    trainer = Trainer(model, train_loader, val_loader, config, output_dir, device)
    if args.smoke_test:
        train_metrics = trainer.train_one_epoch(max_batches=int(args.smoke_batches))
        val_metrics = trainer.validate(max_batches=1)
        trainer.save_checkpoint(output_dir / "smoke_model.pt", epoch=0, is_best=False)
        print(
            json.dumps(
                {
                    "mode": config["mode"],
                    "device": str(device),
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "checkpoint": str(output_dir / "smoke_model.pt"),
                    "test_loader_created": False,
                },
                indent=2,
            )
        )
        train_loader.dataset.close()
        val_loader.dataset.close()
        return 0

    summary = trainer.fit()
    train_loader.dataset.close()
    val_loader.dataset.close()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
