from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "rl_train/results/Exo_Phase_2_steady/trained_models/model_91815936.zip"
DEFAULT_OUTPUT = REPO_ROOT / "tcn/data_generation_v2/output"
TEACHER_HZ = 30
PHYSICS_HZ = 1200
TARGET_HZ = 100
HISTORY_STEPS = 100
INPUT_CHANNELS = (
    "left_thigh_angle_rad",
    "left_thigh_gyro_rad_s",
    "right_thigh_angle_rad",
    "right_thigh_gyro_rad_s",
)
TARGET_CHANNELS = ("left_teacher_action_norm", "right_teacher_action_norm")


@dataclass(frozen=True)
class PipelineConfig:
    checkpoint: Path = DEFAULT_CHECKPOINT
    output_dir: Path = DEFAULT_OUTPUT
    min_velocity: float = 0.90
    max_velocity: float = 1.60
    velocity_step: float = 0.05
    episodes_per_velocity: int = 100
    episode_duration: float = 5.0
    seed: int = 91815936
    overwrite: bool = False
    smoke_test: bool = False
    smoke_episodes_per_velocity: int = 2

    @property
    def raw_path(self) -> Path:
        return self.output_dir / "raw_30hz" / "steady_30hz.h5"

    @property
    def processed_path(self) -> Path:
        return self.output_dir / "processed_100hz" / "steady_100hz.h5"

    @property
    def dataset_dir(self) -> Path:
        return self.output_dir / "dataset"

    def velocities(self) -> list[float]:
        lo = int(round(self.min_velocity * 100))
        hi = int(round(self.max_velocity * 100))
        step = int(round(self.velocity_step * 100))
        if step <= 0:
            raise ValueError("velocity_step must be positive")
        values = list(range(lo, hi + 1, step))
        return [v / 100.0 for v in values]

    def effective_episodes_per_velocity(self) -> int:
        return self.smoke_episodes_per_velocity if self.smoke_test else self.episodes_per_velocity


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    p = argparse.ArgumentParser(description="TCN data generation v2 steady pipeline")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-directory", "--output-dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--minimum-velocity", "--min-velocity", dest="min_velocity", type=float, default=0.90)
    p.add_argument("--maximum-velocity", "--max-velocity", dest="max_velocity", type=float, default=1.60)
    p.add_argument("--velocity-step", type=float, default=0.05)
    p.add_argument("--episodes-per-velocity", type=int, default=100)
    p.add_argument("--episode-duration", type=float, default=5.0)
    p.add_argument("--random-seed", "--seed", dest="seed", type=int, default=91815936)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-episodes-per-velocity", type=int, default=2)
    args = p.parse_args(argv)
    return PipelineConfig(
        checkpoint=args.checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        min_velocity=float(args.min_velocity),
        max_velocity=float(args.max_velocity),
        velocity_step=float(args.velocity_step),
        episodes_per_velocity=int(args.episodes_per_velocity),
        episode_duration=float(args.episode_duration),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
        smoke_test=bool(args.smoke_test),
        smoke_episodes_per_velocity=int(args.smoke_episodes_per_velocity),
    )
