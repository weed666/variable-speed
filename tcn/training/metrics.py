from __future__ import annotations

import torch


def regression_metrics(pred: torch.Tensor, target: torch.Tensor, torque_scale_nm: float) -> dict[str, float]:
    with torch.no_grad():
        err = pred - target
        mse = torch.mean(err * err)
        mae = torch.mean(torch.abs(err))
        rmse = torch.sqrt(mse)
        per_side_mse = torch.mean(err * err, dim=0)
        per_side_mae = torch.mean(torch.abs(err), dim=0)
        return {
            "mse_norm": float(mse.detach().cpu()),
            "rmse_norm": float(rmse.detach().cpu()),
            "mae_norm": float(mae.detach().cpu()),
            "rmse_nm": float((rmse * torque_scale_nm).detach().cpu()),
            "mae_nm": float((mae * torque_scale_nm).detach().cpu()),
            "left_rmse_norm": float(torch.sqrt(per_side_mse[0]).detach().cpu()),
            "right_rmse_norm": float(torch.sqrt(per_side_mse[1]).detach().cpu()),
            "left_mae_norm": float(per_side_mae[0].detach().cpu()),
            "right_mae_norm": float(per_side_mae[1].detach().cpu()),
            "left_rmse_nm": float((torch.sqrt(per_side_mse[0]) * torque_scale_nm).detach().cpu()),
            "right_rmse_nm": float((torch.sqrt(per_side_mse[1]) * torque_scale_nm).detach().cpu()),
            "left_mae_nm": float((per_side_mae[0] * torque_scale_nm).detach().cpu()),
            "right_mae_nm": float((per_side_mae[1] * torque_scale_nm).detach().cpu()),
        }
