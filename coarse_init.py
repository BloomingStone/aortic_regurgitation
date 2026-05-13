"""粗搜索初始化位姿 — 在 alpha/beta/sod 网格上搜索最佳初始参数。"""

from __future__ import annotations

from math import radians
from itertools import product
from typing import Callable

import torch
from torch import Tensor
import torch.nn.functional as F

from tqdm import tqdm

from diffdrr.drr import DRR
from diffdrr.registration import Registration

from common_types import GridSearchConfig
from geometry import pose_from_carm
from drr import project_image


@torch.no_grad()
def coarse_init_pose(
    drr: DRR,
    gt_img: Tensor,
    alpha_grid_config: GridSearchConfig,
    beta_grid_config: GridSearchConfig,
    sod_grid_config: GridSearchConfig,
    loss_fn: Callable,
    img_downsample_stride: int = 4,
    init_gamma_deg: float = 0.0,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """在 alpha、beta、sod 三维网格上搜索最优初始位姿。

    对每个候选参数：
    1. 构造 C-arm 位姿
    2. 投影 DRR
    3. 计算与真实图像的损失
    4. 追踪最优参数

    Args:
        drr: DRR 渲染器。
        gt_img: 真实 X 光图像 (1, 1, H, W) 或 (1, H, W)。
        alpha_grid_config: alpha 角搜索配置。
        beta_grid_config: beta 角搜索配置。
        sod_grid_config: SOD 搜索配置。
        loss_fn: 损失函数。
        img_downsample_stride: 搜索时的图像下采样倍率。
        init_gamma_deg: gamma 角初始值（度）。
        device: 计算设备，默认从 gt_img 推断。

    Returns:
        (best_rotations, best_translations)
    """
    device = gt_img.device if device is None else device

    gt_small = gt_img
    if gt_small.dim() == 3:
        gt_small = gt_small.unsqueeze(0)
    if img_downsample_stride > 1:
        gt_small = F.avg_pool2d(
            gt_small, kernel_size=img_downsample_stride, stride=img_downsample_stride
        )

    alpha_candidates = alpha_grid_config.to_candidates()
    beta_candidates = beta_grid_config.to_candidates()
    sod_candidates = sod_grid_config.to_candidates()

    best_score = float("inf")
    best_alpha_deg: float = 0.0
    best_beta_deg: float = 0.0
    best_sod_mm: float = 0.0
    best_rotations: Tensor | None = None
    best_translations: Tensor | None = None

    pbar = tqdm(
        product(alpha_candidates, beta_candidates, sod_candidates),
        total=len(alpha_candidates) * len(beta_candidates) * len(sod_candidates),
        desc="Coarse grid search",
        ncols=120,
    )
    for alpha_deg, beta_deg, sod_mm in pbar:
        rotations, translations = pose_from_carm(
            sod_mm, 0.0, 0.0,
            radians(alpha_deg), radians(beta_deg), radians(init_gamma_deg),
        )
        reg = Registration(
            drr,
            rotations.to(device),
            translations.to(device),
            parameterization="euler_angles",
            convention="ZXY",
        )
        img = project_image(reg, downsample_stride=img_downsample_stride)
        score = float(loss_fn(img, gt_small).item())
        if score < best_score:
            best_score = score
            best_alpha_deg = alpha_deg
            best_beta_deg = beta_deg
            best_sod_mm = sod_mm
            best_rotations = rotations
            best_translations = translations
        pbar.set_postfix(
            best_score=f"{best_score:.6f}",
            alpha=f"{best_alpha_deg:.2f}",
            beta=f"{best_beta_deg:.2f}",
            sod=f"{best_sod_mm:.2f}",
        )

    assert best_rotations is not None and best_translations is not None
    return best_rotations, best_translations
