"""DRR 构造与投影函数。"""

from __future__ import annotations

from math import radians
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from torchio import LabelMap, ScalarImage, Subject

from diffdrr.drr import DRR
from diffdrr.registration import Registration

from common_types import CArmGeometry, DrrSetting, LabelID, WholeHeartLabelID
from preprocessing import (
    load_nifti,
    get_clip_roi_from_label,
    build_semantic_masks,
    adjust_iodine_contrast,
)
from geometry import get_reorientation, get_label_centering_affine
from transforms import ResampleTransform


def get_drr(
    img_path: Path,
    seg_path: Path,
    whole_heart_label_path: Path,
    geom: CArmGeometry,
    device: torch.device,
    config: DrrSetting = DrrSetting(),
) -> DRR:
    """构造 DRR 渲染器。

    完整流程：加载体数据 → 裁剪 ROI → 重采样 → 重新居中 → 对比度调整 → 构造 DRR。

    Args:
        img_path: CTA 图像路径。
        seg_path: 主动脉/左心室分割路径。
        whole_heart_label_path: 全心分割路径。
        geom: C-arm 几何参数。
        device: 计算设备。
        config: DRR 渲染配置。

    Returns:
        DRR 实例（已移至指定设备）。
    """
    volume = load_nifti(img_path)
    label = load_nifti(seg_path, is_label=True)
    whole_heart_label = load_nifti(whole_heart_label_path, is_label=True)

    volume.to_device(device)
    label.to_device(device)
    whole_heart_label.to_device(device)

    print("Clip ROI Original volume shape:", volume.data.shape)
    clip_roi = get_clip_roi_from_label(
        (whole_heart_label.data == WholeHeartLabelID.HEART),
        margin=5,
        keep_xy_shape=False,
    )
    volume = clip_roi(volume)
    label = clip_roi(label)
    whole_heart_label = clip_roi(whole_heart_label)
    print("Clip ROI Clipped volume shape:", volume.data.shape)
    
    # bone_mask = volume.data > 300
    # clip_roi_bone = get_clip_roi_from_label(bone_mask, margin=10, keep_xy_shape=False)
    # volume = clip_roi_bone(volume)
    # label = clip_roi_bone(label)
    # whole_heart_label = clip_roi_bone(whole_heart_label)
    # print("Clip Bone-ROI volume shape:", volume.data.shape)

    print(f"Resampling by factor {config.resample_factor} to reduce memory usage and speed up training")
    resample = ResampleTransform(resample_factor=config.resample_factor)
    volume.data = volume.data.float()
    volume = resample(volume)
    label = resample(label)
    whole_heart_label = resample(whole_heart_label)
    print("Resampled volume shape:", volume.data.shape)

    print("Re-centering the volume to the center of mass of the whole heart label")
    affine = get_label_centering_affine(
        (whole_heart_label.data > 0), whole_heart_label.affine
    )
    masks = build_semantic_masks(label.data, whole_heart_label.data)
    mapped_volume = adjust_iodine_contrast(volume.data, masks["heart"], masks["aorta"])

    assert mapped_volume.dim() >= 3
    w, h, d = mapped_volume.shape[-3:]
    shape = (1, w, h, d)

    subject = Subject(
        volume=ScalarImage(tensor=mapped_volume.reshape(*shape).float().to(device), affine=affine),
        mask=LabelMap(tensor=label.data.reshape(*shape).float().to(device), affine=affine),
        reorient=get_reorientation("AP"),  # type: ignore
        density=ScalarImage(tensor=mapped_volume.reshape(*shape).float().to(device), affine=affine),
        fiducials=None,  # type: ignore
    )

    drr = DRR(
        subject=subject,
        sdd=geom.sdd,
        height=geom.height,
        width=geom.width,
        delx=geom.delx,
        dely=geom.dely,
        x0=geom.x0,
        y0=geom.y0,
        patch_size=config.patch_size,
        checkpoint_gradients=True,
    ).to(device)

    return drr


def project_image(reg: Registration, downsample_stride: int = 1) -> Tensor:
    """执行投影并返回归一化 DRR 图像。

    注意：DRR 是密度投影，需反转得到 X 光强度图像。

    Args:
        reg: Registration 实例。
        downsample_stride: 下采样步长（>1 时降采样）。

    Returns:
        归一化投影图像 (1, 1, H, W)。
    """
    proj: Tensor = reg(mask_to_channels=True)
    if proj.dim() == 4 and proj.shape[1] > 1:
        img = proj.sum(dim=1, keepdim=True)
    else:
        img = proj if proj.dim() == 4 else proj.unsqueeze(1)
    img = img.max() - img  # 反转：密度 → 强度
    img = img - img.min()
    img = img / img.max().clamp(min=1e-6)

    if downsample_stride > 1:
        img = F.avg_pool2d(img, kernel_size=downsample_stride, stride=downsample_stride)

    return img


@torch.no_grad()
def valid_project_image_label(reg: Registration) -> tuple[Tensor, Tensor]:
    """验证时执行投影并返回图像和标签。

    Returns:
        (img, label): 归一化图像 (1, 1, H, W) 和多类标签 (H, W)。
    """
    proj: Tensor = reg(mask_to_channels=True)
    if proj.dim() == 4 and proj.shape[1] > 1:
        img = proj.sum(dim=1, keepdim=True)
    else:
        img = proj if proj.dim() == 4 else proj.unsqueeze(1)
    img = img.max() - img
    img = img - img.min()
    img = img / img.max().clamp(min=1e-6)

    label = (proj > 0).to(torch.uint8).detach().squeeze()[[0, LabelID.AO, LabelID.LV]]
    label[0] = 0
    label = label.argmax(0)

    return img, label


def run(reg: Registration, downsample_stride: int = 1) -> tuple[Tensor, Tensor]:
    """训练中执行投影，返回图像和标签。

    Returns:
        (img, label): 归一化图像和多类标签。
    """
    img = project_image(reg, downsample_stride=downsample_stride)

    proj: Tensor = reg(mask_to_channels=True)
    label = (proj > 0).to(torch.uint8).detach().squeeze()[[0, LabelID.AO, LabelID.LV]]
    label[0] = 0
    label = label.argmax(0)
    return img, label


def run_image_only(reg: Registration, downsample_stride: int = 1) -> Tensor:
    """仅返回投影图像（不含标签），用于训练。"""
    return project_image(reg, downsample_stride=downsample_stride)


@torch.no_grad()
def run_label_only(reg: Registration) -> Tensor:
    """仅返回投影标签（无梯度）。"""
    proj: Tensor = reg(mask_to_channels=True)
    label = (proj > 0).to(torch.uint8).detach().squeeze()[[0, LabelID.AO, LabelID.LV]]
    label[0] = 0
    return label.argmax(0)


def run_with_masks(reg: Registration) -> tuple[Tensor, Tensor, Tensor]:
    """执行投影，返回图像和分离的 AO/LV 掩码。

    Returns:
        (img, ao_mask, lv_mask): 归一化图像 (H, W)、主动脉掩码 (H, W)、左心室掩码 (H, W)。
    """
    proj: Tensor = reg(mask_to_channels=True)
    img = proj.sum(dim=1, keepdim=True)
    img = img.max() - img
    img = img / img.max()

    masks = (proj > 0).to(torch.float32).detach().squeeze()  # (3, H, W)
    ao_mask = masks[LabelID.AO]
    lv_mask = masks[LabelID.LV]

    del proj
    return img, ao_mask, lv_mask
