from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from .dataset import TCNDataset
from .sampler import UnifiedRegimeBalancedSampler


def create_dataloader(
    mode: str,
    split: str,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    normalize: bool = True,
    history_steps: int = 100,
    unified_balance: bool = True,
    unified_normalization: str = "regime_balanced",
    drop_last: bool = False,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = TCNDataset(
        mode=mode,
        split=split,
        history_steps=history_steps,
        normalize=normalize,
        unified_normalization=unified_normalization,
        **dataset_kwargs,
    )

    sampler = None
    if mode == "unified" and split == "train" and unified_balance:
        sampler = UnifiedRegimeBalancedSampler(dataset, replacement=True)
        effective_shuffle = False
    else:
        effective_shuffle = (split == "train") if shuffle is None else bool(shuffle)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
    )
