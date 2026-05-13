"""损失函数、验证指标与损失工厂。"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from diffdrr.metrics import NormalizedCrossCorrelation2d

from monai.losses.image_dissimilarity import (
    LocalNormalizedCrossCorrelationLoss,
    GlobalMutualInformationLoss,
)
from monai.losses.ssim_loss import SSIMLoss

from common_types import ValidMetricKeys


# ── 内部工具 ────────────────────────────────────────────────────

def _sobel_filters(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3)
    return kernel_x, kernel_y


def _gradient_magnitude(image: Tensor) -> Tensor:
    kernel_x, kernel_y = _sobel_filters(image.device, image.dtype)
    grad_x = F.conv2d(image, kernel_x, padding=1)
    grad_y = F.conv2d(image, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)


def _ncc_2d(pred: Tensor, target: Tensor) -> Tensor:
    pred_centered = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target_centered = target - target.mean(dim=(-2, -1), keepdim=True)
    numerator = (pred_centered * target_centered).mean(dim=(-2, -1))
    denominator = torch.sqrt(
        pred_centered.square().mean(dim=(-2, -1)) *
        target_centered.square().mean(dim=(-2, -1)) + 1e-8
    )
    return numerator / denominator


# ── Checkpointed LNCC ───────────────────────────────────────────

class CheckpointedLNCC(nn.Module):
    """使用梯度检查点的 LNCC 损失，节省显存。"""

    def __init__(self, lncc: LocalNormalizedCrossCorrelationLoss):
        super().__init__()
        self.lncc = lncc

    def forward(self, pred, target):
        return torch.utils.checkpoint.checkpoint(self.lncc, pred, target)  # type: ignore


# ── 验证评分 ────────────────────────────────────────────────────

def validation_score(metrics: dict[ValidMetricKeys, float]) -> float:
    """综合验证评分 = SSIM + MI + NCC。"""
    return (
        + 1.0 * metrics[ValidMetricKeys.SSIM]
        + 1.0 * metrics[ValidMetricKeys.MI]
        + 1.0 * metrics[ValidMetricKeys.NCC]
    )


# ── 损失函数工厂 ────────────────────────────────────────────────

def get_loss_fn(
    cfg: dict[str, Any] | list[dict[str, Any]],
    device: torch.device,
) -> Callable[[Tensor, Tensor], Tensor]:
    """从配置构造（可能加权的）损失函数。

    支持单损失或多损失加权组合。配置格式：
    - dict: {"type": "MI", "weight": 1.0, "init_args": {...}}
    - list: [{"type": "MI", ...}, {"type": "SSIM", ...}]

    Args:
        cfg: 损失配置（dict 或 list[dict]）。
        device: 计算设备。

    Returns:
        调用签名为 (pred, target) -> Tensor 的损失函数。
    """

    def _make_single(type_str: str, init_args: dict[str, Any]) -> Callable:
        type_str = type_str.upper()
        match type_str:
            case "NCC":
                ncc = NormalizedCrossCorrelation2d().to(device)
                return lambda x, y: -ncc(x, y)
            case "LNCC":
                lncc = LocalNormalizedCrossCorrelationLoss(**init_args).to(device)
                return CheckpointedLNCC(lncc)
            case "MI":
                return GlobalMutualInformationLoss(**init_args).to(device)
            case "L1" | "MAE":
                return lambda x, y: F.l1_loss(x, y)
            case "L2" | "MSE":
                return lambda x, y: F.mse_loss(x, y)
            case "SSIM":
                return lambda x, y: SSIMLoss(spatial_dims=2)(x, y)
            case _:
                raise NotImplementedError(f"Unknown loss type: {type_str}")

    if isinstance(cfg, dict):
        cfg = [cfg]

    loss_fns: list[tuple[float, Callable]] = []
    for loss_cfg in cfg:
        loss_type = loss_cfg.get("type", "")
        init_args = loss_cfg.get("init_args", {})
        weight = loss_cfg.get("weight", 1.0)
        loss_fns.append((weight, _make_single(loss_type, init_args)))

    def combined(x: Tensor, y: Tensor) -> Tensor:
        total = 0.0
        for w, fn in loss_fns:
            total += w * fn(x, y)
        return total  # type: ignore[return-value]

    return combined
