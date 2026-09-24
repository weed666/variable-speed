from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from .dataset import REGIME_TO_ID, TCNDataset


class UnifiedRegimeBalancedSampler(Sampler[int]):
    """Window-level 1:1:1 sampler for unified train mode."""

    def __init__(
        self,
        dataset: TCNDataset,
        samples_per_regime: int | None = None,
        replacement: bool = True,
        seed: int = 90210,
    ) -> None:
        if dataset.mode != "unified":
            raise ValueError("UnifiedRegimeBalancedSampler requires mode='unified'")
        if dataset.split != "train":
            raise ValueError("UnifiedRegimeBalancedSampler is intended for unified train only")
        self.dataset = dataset
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        self.regime_ranges = {
            regime: dataset.regime_index_ranges[regime_id]
            for regime, regime_id in REGIME_TO_ID.items()
        }
        available = {regime: end - start for regime, (start, end) in self.regime_ranges.items()}
        self.available_by_regime = available
        if samples_per_regime is None:
            self.samples_per_regime = max(available.values()) if replacement else min(available.values())
        else:
            self.samples_per_regime = int(samples_per_regime)
        if not replacement and self.samples_per_regime > min(available.values()):
            raise ValueError("samples_per_regime exceeds smallest regime without replacement")

    def __len__(self) -> int:
        return self.samples_per_regime * len(self.regime_ranges)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        per_regime: list[torch.Tensor] = []
        for regime in ("steady", "acceleration", "deceleration"):
            start, end = self.regime_ranges[regime]
            count = end - start
            if self.replacement:
                local = torch.randint(0, count, (self.samples_per_regime,), generator=generator)
            else:
                local = torch.randperm(count, generator=generator)[: self.samples_per_regime]
            per_regime.append(local + start)
        stacked = torch.stack(per_regime, dim=1).reshape(-1)
        order = torch.randperm(len(stacked), generator=generator)
        for idx in stacked[order].tolist():
            yield int(idx)

    def epoch_counts_by_regime(self) -> dict[str, int]:
        return {regime: self.samples_per_regime for regime in self.regime_ranges}
