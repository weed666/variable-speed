from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from tcn.data.dataset import INPUT_CHANNELS, TARGET_CHANNELS
from tcn.training.losses import mse_loss
from tcn.training.metrics import regression_metrics


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "calculating..."
    total_seconds = int(round(seconds))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days > 0:
        return f"{days} days {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        config: dict[str, Any],
        output_dir: Path,
        device: torch.device | str,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.torque_scale_nm = float(config["torque_scale_nm"])
        self.max_grad_norm = config.get("max_grad_norm")
        self.amp_enabled = bool(config.get("amp", True)) and self.device.type == "cuda"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.best_val_mse = float("inf")
        self.best_epoch = -1
        self.history: list[dict[str, Any]] = []
        self.show_progress = bool(config.get("show_progress", True))
        self.progress_interval_batches = max(1, int(config.get("progress_interval_batches", 50)))
        self.eta_ema_alpha = float(config.get("eta_ema_alpha", 0.1))
        self.training_start_time: float | None = None
        self.completed_epoch_times: list[float] = []

    def _loader_batch_count(self, loader, max_batches: int | None = None) -> int | None:
        try:
            total = len(loader)
        except TypeError:
            return max_batches
        if max_batches is not None:
            return min(total, int(max_batches))
        return total

    def _should_print_progress(self, batch_num: int, total_batches: int | None) -> bool:
        if batch_num == 1:
            return True
        if total_batches is not None and batch_num == total_batches:
            return True
        return batch_num % self.progress_interval_batches == 0

    def _recent_mean_epoch_time(self) -> float | None:
        recent = self.completed_epoch_times[-5:]
        if not recent:
            return None
        return sum(recent) / len(recent)

    def _eta_max_seconds(
        self,
        epoch_idx: int,
        max_epochs: int,
        current_epoch_remaining_s: float | None,
        current_epoch_total_estimate_s: float | None = None,
    ) -> float | None:
        mean_epoch_time = self._recent_mean_epoch_time()
        remaining_complete_epochs = max(max_epochs - epoch_idx - 1, 0)
        if mean_epoch_time is None:
            if current_epoch_total_estimate_s is None:
                return current_epoch_remaining_s
            current_remaining = current_epoch_remaining_s or 0.0
            return current_remaining + remaining_complete_epochs * current_epoch_total_estimate_s
        current_remaining = current_epoch_remaining_s or 0.0
        return current_remaining + remaining_complete_epochs * mean_epoch_time

    def _print_batch_progress(
        self,
        *,
        epoch_idx: int,
        max_epochs: int,
        phase: str,
        batch_num: int,
        total_batches: int | None,
        elapsed_s: float,
        phase_eta_s: float | None,
        eta_max_s: float | None = None,
        loss: float | None = None,
        lr: float | None = None,
        approximate: bool = False,
    ) -> None:
        if not self.show_progress:
            return
        total_text = str(total_batches) if total_batches is not None else "?"
        parts = [
            f"Epoch {epoch_idx + 1:03d}/{max_epochs:03d}",
            phase,
            f"{batch_num:04d}/{total_text}",
        ]
        if loss is not None:
            parts.append(f"loss={loss:.6f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        parts.append(f"elapsed={format_duration(elapsed_s)}")
        eta_label = "epoch_eta" if phase == "TRAIN" else "val_eta"
        phase_eta = format_duration(phase_eta_s)
        if approximate and phase_eta_s is not None:
            phase_eta = f"{phase_eta} approximate"
        parts.append(f"{eta_label}={phase_eta}")
        if eta_max_s is not None:
            eta_max = format_duration(eta_max_s)
            if approximate:
                eta_max = f"{eta_max} approximate"
            parts.append(f"ETA(max)={eta_max}")
        print(" | ".join(parts), flush=True)

    def _build_optimizer(self):
        if self.config.get("optimizer", "adamw").lower() != "adamw":
            raise ValueError("Only adamw optimizer is supported in the baseline trainer")
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )

    def _build_scheduler(self):
        if self.config.get("scheduler", "reduce_on_plateau") != "reduce_on_plateau":
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(self.config["scheduler_factor"]),
            patience=int(self.config["scheduler_patience"]),
            min_lr=float(self.config["min_learning_rate"]),
        )

    def _autocast(self):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=self.amp_enabled)
        return torch.cuda.amp.autocast(enabled=self.amp_enabled)

    def train_one_epoch(
        self,
        max_batches: int | None = None,
        *,
        epoch_idx: int = 0,
        max_epochs: int = 1,
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        preds = []
        targets = []
        start = time.perf_counter()
        last_batch_time = start
        ema_batch_time: float | None = None
        total_batches = self._loader_batch_count(self.train_loader, max_batches)
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.perf_counter()
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                pred = self.model(x)
                loss = mse_loss(pred, y)
            self.scaler.scale(loss).backward()
            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), float(self.max_grad_norm))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            bs = x.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            total_samples += bs
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
            batch_num = batch_idx + 1
            batch_time = time.perf_counter() - batch_start
            if ema_batch_time is None:
                ema_batch_time = batch_time
            else:
                ema_batch_time = self.eta_ema_alpha * batch_time + (1.0 - self.eta_ema_alpha) * ema_batch_time
            last_batch_time = time.perf_counter()
            if total_batches is not None:
                remaining_batches = max(total_batches - batch_num, 0)
            else:
                remaining_batches = None
            eta_ready = (
                ema_batch_time is not None
                and remaining_batches is not None
                and (batch_num >= 3 or remaining_batches == 0)
            )
            epoch_eta_s = remaining_batches * ema_batch_time if eta_ready else None
            current_epoch_total_estimate_s = (last_batch_time - start) + epoch_eta_s if epoch_eta_s is not None else None
            eta_max_s = self._eta_max_seconds(
                epoch_idx,
                max_epochs,
                epoch_eta_s,
                current_epoch_total_estimate_s=current_epoch_total_estimate_s,
            )
            if self._should_print_progress(batch_num, total_batches):
                running_loss = total_loss / max(total_samples, 1)
                self._print_batch_progress(
                    epoch_idx=epoch_idx,
                    max_epochs=max_epochs,
                    phase="TRAIN",
                    batch_num=batch_num,
                    total_batches=total_batches,
                    elapsed_s=last_batch_time - start,
                    phase_eta_s=epoch_eta_s,
                    eta_max_s=eta_max_s,
                    loss=running_loss,
                    lr=float(self.optimizer.param_groups[0]["lr"]),
                    approximate=self._recent_mean_epoch_time() is None,
                )
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
        metrics = regression_metrics(torch.cat(preds), torch.cat(targets), self.torque_scale_nm)
        metrics["loss"] = total_loss / max(total_samples, 1)
        metrics["epoch_time_s"] = time.perf_counter() - start
        metrics["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    def validate(
        self,
        max_batches: int | None = None,
        *,
        epoch_idx: int = 0,
        max_epochs: int = 1,
        train_time_s: float | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        preds = []
        targets = []
        start = time.perf_counter()
        ema_batch_time: float | None = None
        total_batches = self._loader_batch_count(self.val_loader, max_batches)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch_start = time.perf_counter()
                x = batch["x"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                pred = self.model(x)
                loss = mse_loss(pred, y)
                bs = x.shape[0]
                total_loss += float(loss.detach().cpu()) * bs
                total_samples += bs
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
                batch_num = batch_idx + 1
                batch_time = time.perf_counter() - batch_start
                if ema_batch_time is None:
                    ema_batch_time = batch_time
                else:
                    ema_batch_time = self.eta_ema_alpha * batch_time + (1.0 - self.eta_ema_alpha) * ema_batch_time
                if total_batches is not None:
                    remaining_batches = max(total_batches - batch_num, 0)
                else:
                    remaining_batches = None
                eta_ready = (
                    ema_batch_time is not None
                    and remaining_batches is not None
                    and (batch_num >= 3 or remaining_batches == 0)
                )
                val_eta_s = remaining_batches * ema_batch_time if eta_ready else None
                current_epoch_remaining = val_eta_s
                current_epoch_total_estimate_s = None
                if train_time_s is not None and val_eta_s is not None:
                    current_epoch_total_estimate_s = train_time_s + (time.perf_counter() - start) + val_eta_s
                eta_max_s = self._eta_max_seconds(
                    epoch_idx,
                    max_epochs,
                    current_epoch_remaining,
                    current_epoch_total_estimate_s=current_epoch_total_estimate_s,
                )
                if self._should_print_progress(batch_num, total_batches):
                    self._print_batch_progress(
                        epoch_idx=epoch_idx,
                        max_epochs=max_epochs,
                        phase="VAL",
                        batch_num=batch_num,
                        total_batches=total_batches,
                        elapsed_s=time.perf_counter() - start,
                        phase_eta_s=val_eta_s,
                        eta_max_s=eta_max_s,
                        approximate=self._recent_mean_epoch_time() is None,
                    )
                if max_batches is not None and batch_idx + 1 >= max_batches:
                    break
        metrics = regression_metrics(torch.cat(preds), torch.cat(targets), self.torque_scale_nm)
        metrics["loss"] = total_loss / max(total_samples, 1)
        metrics["val_time_s"] = time.perf_counter() - start
        return metrics

    def fit(self, max_epochs: int | None = None) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        max_epochs = int(max_epochs or self.config["max_epochs"])
        patience = int(self.config["early_stopping_patience"])
        stale = 0
        early_stopped = False
        self.training_start_time = time.perf_counter()
        for epoch in range(max_epochs):
            epoch_start = time.perf_counter()
            lr_before_scheduler = float(self.optimizer.param_groups[0]["lr"])
            train = self.train_one_epoch(epoch_idx=epoch, max_epochs=max_epochs)
            val = self.validate(epoch_idx=epoch, max_epochs=max_epochs, train_time_s=train["epoch_time_s"])
            if self.scheduler is not None:
                self.scheduler.step(val["mse_norm"])
            lr_after_scheduler = float(self.optimizer.param_groups[0]["lr"])
            epoch_time_s = time.perf_counter() - epoch_start
            self.completed_epoch_times.append(epoch_time_s)
            elapsed_time_s = time.perf_counter() - self.training_start_time
            estimated_remaining_s = self._eta_max_seconds(epoch, max_epochs, 0.0)
            row = {
                "epoch": epoch,
                "train_loss": train["loss"],
                "val_loss": val["loss"],
                "train_rmse_norm": train["rmse_norm"],
                "val_rmse_norm": val["rmse_norm"],
                "train_rmse_nm": train["rmse_nm"],
                "val_rmse_nm": val["rmse_nm"],
                "train_left_rmse_norm": train["left_rmse_norm"],
                "train_right_rmse_norm": train["right_rmse_norm"],
                "val_left_rmse_norm": val["left_rmse_norm"],
                "val_right_rmse_norm": val["right_rmse_norm"],
                "learning_rate": train["learning_rate"],
                "epoch_time_s": epoch_time_s,
                "train_time_s": train["epoch_time_s"],
                "val_time_s": val["val_time_s"],
                "elapsed_time_s": elapsed_time_s,
                "estimated_remaining_s": estimated_remaining_s,
            }
            self.history.append(row)
            self._write_log()
            is_best = val["mse_norm"] < self.best_val_mse
            if is_best:
                self.best_val_mse = val["mse_norm"]
                self.best_epoch = epoch
                stale = 0
                self.save_checkpoint(self.output_dir / "best_model.pt", epoch, is_best=True)
                early_stopping_text = "Early stopping: reset (new best)"
            else:
                stale += 1
                early_stopping_text = f"Early stopping: {stale}/{patience}"
            self.save_checkpoint(self.output_dir / "last_model.pt", epoch, is_best=False)
            print(
                "\n".join(
                    [
                        f"Epoch {epoch + 1:03d}/{max_epochs:03d} COMPLETE",
                        f"train_mse={train['mse_norm']:.8f}",
                        f"val_mse={val['mse_norm']:.8f}",
                        f"train_rmse_norm={train['rmse_norm']:.8f}",
                        f"val_rmse_norm={val['rmse_norm']:.8f}",
                        f"lr={lr_after_scheduler:.2e}",
                        f"epoch_time={format_duration(epoch_time_s)}",
                        f"elapsed={format_duration(elapsed_time_s)}",
                        f"ETA(max)={format_duration(estimated_remaining_s)}",
                        f"best_val_mse={self.best_val_mse:.8f}",
                        f"best_epoch={self.best_epoch + 1}",
                        early_stopping_text,
                    ]
                ),
                flush=True,
            )
            if lr_after_scheduler < lr_before_scheduler:
                print(f"LR reduced: {lr_before_scheduler:.2e} -> {lr_after_scheduler:.2e}", flush=True)
            if stale >= patience:
                early_stopped = True
                print(
                    "\n".join(
                        [
                            f"Early stopping triggered at epoch {epoch + 1}.",
                            f"Best epoch = {self.best_epoch + 1}",
                            f"Best val MSE = {self.best_val_mse:.8f}",
                            f"Total training time = {format_duration(time.perf_counter() - self.training_start_time)}",
                        ]
                    ),
                    flush=True,
                )
                break
        total_elapsed_s = time.perf_counter() - self.training_start_time
        summary = {
            "mode": self.config["mode"],
            "best_epoch": self.best_epoch,
            "best_val_mse": self.best_val_mse,
            "epochs_completed": len(self.history),
            "total_elapsed_s": total_elapsed_s,
            "best_checkpoint_path": str(self.output_dir / "best_model.pt"),
            "last_checkpoint_path": str(self.output_dir / "last_model.pt"),
            "early_stopped": early_stopped,
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            "\n".join(
                [
                    "Training complete",
                    f"mode = {summary['mode']}",
                    f"best epoch = {self.best_epoch + 1 if self.best_epoch >= 0 else 'none'}",
                    f"best val MSE = {self.best_val_mse:.8f}",
                    f"epochs completed = {len(self.history)}",
                    f"total elapsed = {format_duration(total_elapsed_s)}",
                    f"best checkpoint path = {self.output_dir / 'best_model.pt'}",
                    f"last checkpoint path = {self.output_dir / 'last_model.pt'}",
                    f"early stopped = {str(early_stopped).lower()}",
                ]
            ),
            flush=True,
        )
        return summary

    def _write_log(self) -> None:
        path = self.output_dir / "training_log.csv"
        if not self.history:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

    def checkpoint_payload(self, epoch: int, is_best: bool) -> dict[str, Any]:
        ds = self.train_loader.dataset
        return {
            "model_type": "causal_tcn",
            "mode": self.config["mode"],
            "state_dict": self.model.state_dict(),
            "model_config": self.model.model_config_dict(),
            "sensor_hz": int(self.config["sensor_hz"]),
            "history_steps": int(self.config["history_steps"]),
            "input_channel_names": list(INPUT_CHANNELS),
            "target_channel_names": list(TARGET_CHANNELS),
            "input_mean": ds.mean.tolist(),
            "input_std": ds.std.tolist(),
            "normalization_scheme": ds.normalization_scheme,
            "torque_scale_nm": self.torque_scale_nm,
            "training_config": self.config,
            "best_epoch": self.best_epoch,
            "best_val_mse": self.best_val_mse,
            "epoch": int(epoch),
            "is_best": bool(is_best),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "dataset_provenance": {
                "raw_files": sorted({str(info.h5_path) for info in ds.episode_infos}),
                "split": ds.split,
                "mode": ds.mode,
                "episode_count": len(ds.episode_infos),
                "window_count": len(ds),
            },
        }

    def save_checkpoint(self, path: Path, epoch: int, is_best: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(epoch, is_best), path)

    @staticmethod
    def cleanup_output_dir(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
