"""体数据加载、裁剪、对比度调整等预处理函数。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.utils.dlpack import from_dlpack as dlpack2tensor
from torch.utils.dlpack import to_dlpack as tensor2dlpack  # type: ignore

import nibabel as nib
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image

from transforms import MedicalImage, ClipROITransform, AABB, ResampleTransform
from common_types import LabelID, WholeHeartLabelID


def pre_load(file: Path) -> tuple[Nifti1Image, ndarray]:
    """加载 NIfTI 文件，返回 (nifti_image, affine_matrix)。"""
    image_nii = nib_load(file)
    assert isinstance(image_nii, Nifti1Image)
    affine = image_nii.affine if image_nii.affine is not None else np.eye(4)
    return image_nii, affine


def load_nifti(file: Path, is_label: bool = False) -> MedicalImage:
    """加载 NIfTI 文件为 MedicalImage（tensor shape: 1,1,D,H,W）。"""
    img, affine = pre_load(file)
    tensor = torch.from_numpy(img.get_fdata())
    if is_label:
        tensor = tensor.round().to(torch.uint8)
    else:
        tensor = tensor.to(torch.float32)
    tensor = tensor[None][None]  # 添加 batch 和 channel 维度
    return MedicalImage(data=tensor, affine=affine)


def get_clip_roi_from_label(
    label: Tensor,
    margin: int | tuple[int, int, int] = (30, 30, 5),
    keep_xy_shape: bool = True
) -> ClipROITransform:
    """根据标签边界框生成 ROI 裁剪变换。

    Args:
        label: 二值标签 (1, 1, D, H, W)。
        margin: 边界框边距（体素）。
        keep_xy_shape: 是否保留原始 x, y 尺寸（仅 z 方向裁剪）。

    Returns:
        ClipROITransform 实例。
    """
    if isinstance(margin, int):
        margin = (margin, margin, margin)
    
    if any([m < 0 for m in margin]):
        raise ValueError("Margin must be non-negative")
    if label.dim() < 3:
        raise ValueError("Label must have at least 3 dimensions (D, H, W)")
    shape = torch.tensor(label.shape[-3:])
    
    margin_tensor = torch.tensor(margin)

    label_bin = label.squeeze() > 0
    coords = torch.nonzero(label_bin)
    min_coords = coords.min(dim=0).values - margin_tensor
    max_coords = coords.max(dim=0).values + margin_tensor

    min_coords = torch.max(min_coords, torch.zeros_like(min_coords))
    max_coords = torch.min(max_coords, shape - 1)

    if keep_xy_shape:
        min_coords[:2] = 0
        max_coords[:2] = shape[:2]

    return ClipROITransform(AABB(
        x_min=int(min_coords[0]), x_max=int(max_coords[0]),
        y_min=int(min_coords[1]), y_max=int(max_coords[1]),
        z_min=int(min_coords[2]), z_max=int(max_coords[2])
    ))


def build_semantic_masks(
    ascending_aorta_LV_label: Tensor,
    whole_heart_label: Tensor
) -> dict[Literal["ascending_aorta", "lv", "CTA_contrast_area"], Tensor]:
    """构建语义掩码字典。

    Returns:
        {"ascending_aorta": 主动脉掩码, "lv": 左心室掩码, "CTA_contrast_area": CTA 对比度区域掩码}
    """
    return {
        "ascending_aorta": (ascending_aorta_LV_label == LabelID.AO),
        "lv": ascending_aorta_LV_label == LabelID.LV,
        "CTA_contrast_area": (whole_heart_label > 0) | (ascending_aorta_LV_label > 0),
    }


def adjust_iodine_contrast(
    volume: Tensor,
    label_to_water: Tensor,
    label_contrast: Tensor,
    contrast_HU: float = 200.0
) -> Tensor:
    """调整 CTA 碘造影剂对比度。

    Args:
        volume: 输入 CTA 体数据 (1, 1, D, H, W)。
        label_to_water: 需映射到 HU=0 的区域掩码。
        label_contrast: 需增强对比度的区域掩码（主动脉）。
        contrast_HU: 造影剂增强量 (HU)。

    Returns:
        归一化到 [0, 1] 的体数据。
    """
    res = volume.clone()

    # 体外仪器
    res[res > 1500] = res.min()

    # 骨骼区域
    res[res > 300] *= 0.5

    # CTA 造影剂区域 + label_to_water 区域 → 水（HU ≈ 0）
    threshold_mask = (volume > 100) & (volume < 300)
    water_mask = (label_to_water > 0) | threshold_mask 
    res[water_mask] = 0.
    
    # 主动脉造影剂增强
    res[label_contrast > 0] += contrast_HU
    
    # HU → 相对值，clip 到 [0, 2]（组织 HU > -200）
    res = ((res + 200) / 1000).clamp(min=0., max=2.)



    # normalize to [0, 1]
    res -= res.min()
    res_max = res.max()
    res /= res_max

    return res
