"""可视化 — 图像保存与标签叠加。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from PIL import Image


def save_label_overlay(
    gt_img: Tensor,
    ao_mask: Tensor,
    lv_mask: Tensor,
    output_path: Path,
    alpha: float = 0.7,
) -> None:
    """将 AO/LV 标签叠加到真实图像上保存。

    AO: 红色，LV: 绿色。alpha 控制掩码透明度 (0-1)。

    Args:
        gt_img: 灰度真实图像 (H, W) 或 (1, H, W)。
        ao_mask: 主动脉掩码 (H, W)。
        lv_mask: 左心室掩码 (H, W)。
        output_path: 输出 PNG 路径。
        alpha: 掩码透明度，默认 0.7。
    """
    gt_np = (gt_img.squeeze().cpu().detach() * 255).to(torch.uint8).numpy()
    ao_np = (ao_mask.squeeze().cpu().detach().numpy() > 0).astype(np.uint8) * 255
    lv_np = (lv_mask.squeeze().cpu().detach().numpy() > 0).astype(np.uint8) * 255

    rgb = np.stack([gt_np, gt_np, gt_np], axis=-1).astype(np.float32)

    # AO → 红色
    ao_bool = ao_np > 0
    rgb[ao_bool, 0] = rgb[ao_bool, 0] * (1 - alpha) + 255 * alpha
    rgb[ao_bool, 1] = rgb[ao_bool, 1] * (1 - alpha) + 0 * alpha
    rgb[ao_bool, 2] = rgb[ao_bool, 2] * (1 - alpha) + 0 * alpha

    # LV → 绿色
    lv_bool = lv_np > 0
    rgb[lv_bool, 0] = rgb[lv_bool, 0] * (1 - alpha) + 0 * alpha
    rgb[lv_bool, 1] = rgb[lv_bool, 1] * (1 - alpha) + 255 * alpha
    rgb[lv_bool, 2] = rgb[lv_bool, 2] * (1 - alpha) + 0 * alpha

    Image.fromarray(rgb.astype(np.uint8)).save(output_path)


def save_masks_single_channel(
    ao_mask: Tensor,
    lv_mask: Tensor,
    output_stem: Path,
) -> dict[str, Path]:
    """保存单通道掩码图像 (PNG, 值 0 或 255)。

    保存两张图像：
      - {output_stem}_ao.png: AO 二值掩码 (0, 255)
      - {output_stem}_lv.png: LV 二值掩码 (0, 255)

    Args:
        ao_mask: 主动脉掩码 (H, W)。
        lv_mask: 左心室掩码 (H, W)。
        output_stem: 输出路径前缀（不含扩展名）。

    Returns:
        保存的文件路径字典，键为 'ao', 'lv'。
    """
    ao_np = (ao_mask.squeeze().cpu().detach().numpy() > 0).astype(np.uint8) * 255
    lv_np = (lv_mask.squeeze().cpu().detach().numpy() > 0).astype(np.uint8) * 255

    parent = output_stem.parent
    stem = output_stem.stem

    paths: dict[str, Path] = {}

    # AO 单通道 (0, 255)
    p = parent / f"{stem}_ao.png"
    Image.fromarray(ao_np).save(p)
    paths["ao"] = p

    # LV 单通道 (0, 255)
    p = parent / f"{stem}_lv.png"
    Image.fromarray(lv_np).save(p)
    paths["lv"] = p

    return paths


def save_masks_only(
    ao_mask: Tensor,
    lv_mask: Tensor,
    output_path: Path,
    save_single_channel: bool = False,
) -> dict[str, Path] | None:
    """保存纯掩码图像（黑色背景，AO 红色，LV 绿色）。

    同时可选择保存单通道黑白版本（AO、LV 各一张二值图，以及一张组合图）。

    Args:
        ao_mask: 主动脉掩码 (H, W)。
        lv_mask: 左心室掩码 (H, W)。
        output_path: 输出 PNG 路径（三通道彩色）。
        save_single_channel: 是否额外保存单通道版本，默认 False。

    Returns:
        如果 save_single_channel 为 True, 返回单通道文件路径字典；否则返回 None。
    """
    ao_np = (ao_mask.squeeze().cpu().detach().numpy() > 0).astype(np.uint8)
    lv_np = (lv_mask.squeeze().cpu().detach().numpy() > 0).astype(np.uint8)

    h, w = ao_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[ao_np > 0, 0] = 255  # AO 红色
    rgb[lv_np > 0, 1] = 255  # LV 绿色

    Image.fromarray(rgb).save(output_path)

    if save_single_channel:
        stem = output_path.with_suffix("")  # 去掉扩展名作为前缀
        return save_masks_single_channel(ao_mask, lv_mask, stem)
    return None
