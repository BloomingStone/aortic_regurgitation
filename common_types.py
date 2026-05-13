"""基础类型定义：类型别名、枚举、dataclass。

所有模块共享的类型定义集中于此，避免循环导入。
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar
from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np

# ── 物理量类型别名 ──────────────────────────────────────────────
MM = float | int                     # 毫米
Pixel = int                          # 像素
MMPerPixel = float | int             # mm/pixel 像素间距

Degree = float | int                 # 度
DegreePerSec = float | int           # 度/秒
Radian = float | int                 # 弧度
Angle = TypeVar("Angle", Degree, Radian)
type Rot[Angle] = tuple[Angle, Angle, Angle]     # 三元旋转角

Sec = float | int                    # 秒
Point = tuple[float, float, float]   # 3D 点


# ── 标签枚举 ────────────────────────────────────────────────────
class LabelID(IntEnum):
    BACKGROUND = 0
    AO = 1
    LV = 2


class WholeHeartLabelID(IntEnum):
    BACKGROUND = 0
    AO = 1
    HEART = 2


class ValidMetricKeys(StrEnum):
    MSE = "val/mse"
    MAE = "val/mae"
    NCC = "val/ncc"
    PSNR = "val/psnr"
    GRAD_MSE = "val/grad_mse"
    SSIM = "val/ssim"
    MI = "val/mi"


# ── 数据结构 ────────────────────────────────────────────────────
@dataclass
class CArmGeometry:
    """C-arm 成像几何参数"""
    sdd: MM                         # 源到探测器距离 (mm)
    sod: MM                         # 源到物体距离 (mm)
    height: Pixel                   # 图像高度 (pixel)
    delx: MMPerPixel                # x 方向像素间距 (mm/pixel)
    _width: Pixel | None = None     # 图像宽度，默认等于 height
    _dely: MMPerPixel | None = None # y 方向像素间距，默认等于 delx
    x0: MM = 0.0                    # 探测器主点 x 偏移
    y0: MM = 0.0                    # 探测器主点 y 偏移

    def __post_init__(self):
        if self._width is None:
            self._width = self.height
        if self._dely is None:
            self._dely = self.delx

    @property
    def width(self) -> Pixel:
        assert self._width is not None
        return self._width

    @property
    def dely(self) -> MMPerPixel:
        assert self._dely is not None
        return self._dely


@dataclass
class DrrSetting:
    """DRR 渲染参数"""
    patch_size: int | None = None
    orientation_type: Literal["AP", "PA"] | None = "AP"
    parameterization: str = "euler_angles"
    convention: str = "ZXY"
    mask_to_channels: bool = True
    resample_factor: float = 0.5


@dataclass
class GridSearchConfig:
    """粗搜索网格参数"""
    init_value: float
    window_size: float
    step_size: float

    def to_candidates(self) -> list[float]:
        return list(np.arange(
            self.init_value - self.window_size,
            self.init_value + self.window_size + 1e-6,
            self.step_size
        ))


@dataclass
class StageConfig:
    """训练阶段配置"""
    name: str
    downsample_stride: int = 1
    n_itrs: int = 0
    lr_rot_mult: float = 1.0
    lr_trans_mult: float = 1.0

    def __post_init__(self):
        if self.downsample_stride < 1:
            raise ValueError("downsample_stride must be >= 1")
        if self.n_itrs < 0:
            raise ValueError("n_itrs must be >= 0")
        if self.lr_rot_mult <= 0:
            raise ValueError("lr_rot_mult must be > 0")
        if self.lr_trans_mult <= 0:
            raise ValueError("lr_trans_mult must be > 0")
        if self.n_itrs == 0:
            import warnings
            warnings.warn(f"Stage {self.name} has 0 iterations, it will be skipped.")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StageConfig":
        return StageConfig(
            name=d.get("name", "stage"),
            downsample_stride=int(d.get("downsample_stride", 1)),
            n_itrs=int(d.get("n_itrs", 0)),
            lr_rot_mult=float(d.get("lr_rot_mult", 1.0)),
            lr_trans_mult=float(d.get("lr_trans_mult", 1.0)),
        )
