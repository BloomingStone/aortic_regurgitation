"""几何 / 位姿 / 仿射变换工具函数。"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.utils.dlpack import to_dlpack as tensor2dlpack  # type: ignore
from scipy.ndimage import center_of_mass

from diffdrr.drr import convert

from common_types import MM, Radian


def get_reorientation(
    orientation_type: Optional[Literal["AP", "PA"]] = "AP"
) -> torch.Tensor:
    """获取 C-arm 坐标系重定向矩阵（4x4）"""
    if orientation_type == "AP":
        reorient = torch.tensor(
            [[1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]],
            dtype=torch.float32,
        )
    elif orientation_type == "PA":
        reorient = torch.tensor(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]],
            dtype=torch.float32,
        )
    elif orientation_type is None:
        reorient = torch.tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unrecognized orientation {orientation_type}")
    return reorient


def recenter(
    original_affine: ndarray,
    center_voxel: tuple[int, ...] | np.ndarray
) -> ndarray:
    """计算以给定体素为中心的仿射矩阵。

    Args:
        original_affine: 原始 4x4 仿射矩阵。
        center_voxel: 目标中心体素坐标。

    Returns:
        新的 4x4 仿射矩阵，原点位于 center_voxel 处。
    """
    B = original_affine[:3, :3]
    new_t = -(B @ np.array(center_voxel))
    T = np.eye(4)
    T[:3, :3] = B
    T[:3, 3] = new_t
    return T


def get_label_centering_affine(label: Tensor, volume_affine: ndarray) -> ndarray:
    """根据标签质心计算居中仿射矩阵（scipy 版）。

    Args:
        label: 二值标签 (1, 1, D, H, W)。
        volume_affine: 原始 4x4 仿射矩阵。

    Returns:
        新的 4x4 仿射矩阵。
    """
    label_center = center_of_mass((label > 0).squeeze().cpu().numpy())  # type: ignore
    return recenter(volume_affine, label_center)  # type: ignore


def get_coronary_centering_affine(label: Tensor, volume_affine: ndarray) -> ndarray:
    """根据冠脉标签质心计算居中仿射矩阵（cupy 版，用于 GPU 加速）。

    排除 LabelID.AO（主动脉）区域，仅计算心脏其余部分的质心。

    Args:
        label: 标签 (1, 1, D, H, W)，包含 LabelID 枚举值。
        volume_affine: 原始 4x4 仿射矩阵。

    Returns:
        新的 4x4 仿射矩阵。
    """
    from common_types import LabelID as _LabelID
    import cupy as cp
    from cupyx.scipy.ndimage import center_of_mass as cp_center_of_mass

    ref_label = (label > 0) & (label != _LabelID.AO)
    with cp.cuda.Device(label.device.index):
        label_center = cp_center_of_mass(
            cp.from_dlpack(tensor2dlpack(ref_label.squeeze().to(label.device)))
        )
        label_center_tuple = (int(label_center[0]), int(label_center[1]), int(label_center[2]))
        return recenter(volume_affine, label_center_tuple)


def pose_from_carm(
    sod: MM,
    tx: MM,
    ty: MM,
    alpha: Radian,
    beta: Radian,
    gamma: Radian
) -> tuple[Tensor, Tensor]:
    """从 C-arm 几何参数构造位姿（旋转 + 平移）。

    Args:
        sod: 源到物体距离 (mm)。
        tx: x 方向平移 (mm)。
        ty: y 方向平移 (mm)。
        alpha, beta, gamma: Euler 角 (弧度)，按 ZXY 约定。

    Returns:
        (rotations, translations) 两个 (1, 3) 张量。
    """
    rot = torch.tensor([[alpha, beta, gamma]]).float()
    xyz = torch.tensor([[tx, sod, ty]]).float()
    pose = convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
    rotations, translations = pose.convert("euler_angles", "ZXY")
    return rotations, translations
