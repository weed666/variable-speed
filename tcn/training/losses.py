from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)
