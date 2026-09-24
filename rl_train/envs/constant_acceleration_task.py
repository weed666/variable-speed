from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_SPEED_LEVELS = (0.80, 0.95, 1.10, 1.25, 1.40, 1.55, 1.70)
DEFAULT_ACCELERATION_MAGNITUDES = (0.1, 0.2, 0.3, 0.4, 0.5)
VALID_TASK_SAMPLING_STRATEGIES = ("random", "ordered_cycle", "shuffled_cycle")
_DURATION_TOL = 1e-9


@dataclass(frozen=True)
class ConstantAccelerationTask:
    initial_velocity: float
    goal_velocity: float
    signed_acceleration: float
    ramp_duration: float
    delta_velocity: float | None = None
    tuple_acceleration_magnitude: float | None = None

    def as_dict(self) -> dict:
        return {
            "initial_velocity": self.initial_velocity,
            "goal_velocity": self.goal_velocity,
            "signed_acceleration": self.signed_acceleration,
            "ramp_duration": self.ramp_duration,
            "delta_velocity": self.delta_velocity,
            "tuple_acceleration_magnitude": self.tuple_acceleration_magnitude,
        }


@dataclass(frozen=True)
class AccelerationDeltaVTaskTuple:
    acceleration_magnitude: float
    delta_velocity: float
    ramp_duration: float

    def as_dict(self) -> dict:
        return {
            "acceleration_magnitude": self.acceleration_magnitude,
            "delta_velocity": self.delta_velocity,
            "ramp_duration": self.ramp_duration,
        }


def _as_finite_float_list(values: Sequence[float], name: str) -> list[float]:
    if values is None:
        raise ValueError(f"{name} must not be None")
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def build_constant_acceleration_tasks(
    *,
    speed_levels: Sequence[float] = DEFAULT_SPEED_LEVELS,
    acceleration_magnitudes: Sequence[float] = DEFAULT_ACCELERATION_MAGNITUDES,
    min_ramp_duration: float = 1.0,
    max_ramp_duration: float = 5.0,
) -> list[ConstantAccelerationTask]:
    speeds = _as_finite_float_list(speed_levels, "speed_levels")
    accelerations = _as_finite_float_list(acceleration_magnitudes, "acceleration_magnitudes")
    min_duration = float(min_ramp_duration)
    max_duration = float(max_ramp_duration)

    if min_duration <= 0 or max_duration <= 0:
        raise ValueError("ramp duration bounds must be positive")
    if min_duration > max_duration:
        raise ValueError("min_ramp_duration must be <= max_ramp_duration")
    if any(acc <= 0 for acc in accelerations):
        raise ValueError("acceleration_magnitudes must all be positive")
    if len(set(speeds)) != len(speeds):
        raise ValueError("speed_levels must not contain duplicates")

    tasks: list[ConstantAccelerationTask] = []
    for initial_velocity in speeds:
        for goal_velocity in speeds:
            if np.isclose(initial_velocity, goal_velocity, atol=_DURATION_TOL, rtol=0):
                continue
            direction = 1.0 if goal_velocity > initial_velocity else -1.0
            delta_v = abs(goal_velocity - initial_velocity)
            for acceleration_magnitude in accelerations:
                ramp_duration = delta_v / acceleration_magnitude
                if ramp_duration + _DURATION_TOL < min_duration:
                    continue
                if ramp_duration - _DURATION_TOL > max_duration:
                    continue
                tasks.append(
                    ConstantAccelerationTask(
                        initial_velocity=float(initial_velocity),
                        goal_velocity=float(goal_velocity),
                        signed_acceleration=float(direction * acceleration_magnitude),
                        ramp_duration=float(ramp_duration),
                    )
                )
    return tasks


def build_acceleration_delta_v_task_tuples(
    *,
    acceleration_magnitudes: Sequence[float] = DEFAULT_ACCELERATION_MAGNITUDES,
    delta_velocity_levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    min_ramp_duration: float = 1.0,
    max_ramp_duration: float = 5.0,
    tolerance: float = 1e-9,
) -> list[AccelerationDeltaVTaskTuple]:
    accelerations = _as_finite_float_list(acceleration_magnitudes, "acceleration_magnitudes")
    delta_velocities = _as_finite_float_list(delta_velocity_levels, "delta_velocity_levels")
    min_duration = float(min_ramp_duration)
    max_duration = float(max_ramp_duration)

    if min_duration <= 0 or max_duration <= 0:
        raise ValueError("ramp duration bounds must be positive")
    if min_duration > max_duration:
        raise ValueError("min_ramp_duration must be <= max_ramp_duration")
    if any(acc <= 0 for acc in accelerations):
        raise ValueError("acceleration_magnitudes must all be positive")
    if any(delta_v <= 0 for delta_v in delta_velocities):
        raise ValueError("delta_velocity_levels must all be positive")

    task_tuples: list[AccelerationDeltaVTaskTuple] = []
    for acceleration_magnitude in accelerations:
        for delta_velocity in delta_velocities:
            ramp_duration = float(delta_velocity) / float(acceleration_magnitude)
            if ramp_duration + tolerance < min_duration:
                continue
            if ramp_duration - tolerance > max_duration:
                continue
            task_tuples.append(
                AccelerationDeltaVTaskTuple(
                    acceleration_magnitude=float(acceleration_magnitude),
                    delta_velocity=float(delta_velocity),
                    ramp_duration=float(ramp_duration),
                )
            )
    return task_tuples


class ConstantAccelerationTaskSampler:
    def __init__(
        self,
        tasks: Sequence[ConstantAccelerationTask],
        *,
        strategy: str,
        rng,
    ):
        if strategy not in VALID_TASK_SAMPLING_STRATEGIES:
            raise ValueError(
                "task_sampling_strategy must be one of "
                f"{VALID_TASK_SAMPLING_STRATEGIES}, got {strategy!r}"
            )
        self.tasks = list(tasks)
        if not self.tasks:
            raise ValueError("constant acceleration task pool is empty")
        self.strategy = strategy
        self.rng = rng
        self.task_order: list[int] = []
        self.task_cursor = 0
        self._reset_order()

    def _permutation(self, n: int) -> list[int]:
        if hasattr(self.rng, "permutation"):
            return [int(idx) for idx in self.rng.permutation(n)]
        order = list(range(n))
        # Fallback for legacy RandomState-like wrappers.
        for idx in range(n - 1, 0, -1):
            swap_idx = int(self.rng.integers(0, idx + 1))
            order[idx], order[swap_idx] = order[swap_idx], order[idx]
        return order

    def _reset_order(self) -> None:
        if self.strategy == "shuffled_cycle":
            self.task_order = self._permutation(len(self.tasks))
        else:
            self.task_order = list(range(len(self.tasks)))
        self.task_cursor = 0

    def sample(self) -> ConstantAccelerationTask:
        if self.strategy == "random":
            task_index = int(self.rng.integers(0, len(self.tasks)))
            return self.tasks[task_index]

        if self.task_cursor >= len(self.task_order):
            self._reset_order()

        task_index = self.task_order[self.task_cursor]
        self.task_cursor += 1
        return self.tasks[task_index]

    def state_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "task_order": list(self.task_order),
            "task_cursor": int(self.task_cursor),
            "num_tasks": len(self.tasks),
        }


class AccelerationDeltaVTaskTupleSampler:
    def __init__(
        self,
        task_tuples: Sequence[AccelerationDeltaVTaskTuple],
        *,
        strategy: str,
        rng,
    ):
        if strategy not in VALID_TASK_SAMPLING_STRATEGIES:
            raise ValueError(
                "task_sampling_strategy must be one of "
                f"{VALID_TASK_SAMPLING_STRATEGIES}, got {strategy!r}"
            )
        self.task_tuples = list(task_tuples)
        if not self.task_tuples:
            raise ValueError("acceleration delta-v task tuple pool is empty")
        self.strategy = strategy
        self.rng = rng
        self.task_order: list[int] = []
        self.task_cursor = 0
        self._reset_order()

    def _permutation(self, n: int) -> list[int]:
        if hasattr(self.rng, "permutation"):
            return [int(idx) for idx in self.rng.permutation(n)]
        order = list(range(n))
        for idx in range(n - 1, 0, -1):
            swap_idx = int(self.rng.integers(0, idx + 1))
            order[idx], order[swap_idx] = order[swap_idx], order[idx]
        return order

    def _reset_order(self) -> None:
        if self.strategy == "shuffled_cycle":
            self.task_order = self._permutation(len(self.task_tuples))
        else:
            self.task_order = list(range(len(self.task_tuples)))
        self.task_cursor = 0

    def sample_tuple(self) -> AccelerationDeltaVTaskTuple:
        if self.strategy == "random":
            task_index = int(self.rng.integers(0, len(self.task_tuples)))
            return self.task_tuples[task_index]

        if self.task_cursor >= len(self.task_order):
            self._reset_order()

        task_index = self.task_order[self.task_cursor]
        self.task_cursor += 1
        return self.task_tuples[task_index]

    def state_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "task_order": list(self.task_order),
            "task_cursor": int(self.task_cursor),
            "num_task_tuples": len(self.task_tuples),
        }
