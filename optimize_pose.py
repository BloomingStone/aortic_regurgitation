from typing import Literal, Optional, TypeVar, Any, Callable
from dataclasses import dataclass
from math import radians
from pathlib import Path
from enum import IntEnum, StrEnum
from itertools import product
from functools import partial

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
from torch import nn
from torch.utils.dlpack import to_dlpack as tensor2dlpack   #type: ignore
import torch.nn.functional as F
from torchio import LabelMap, ScalarImage, Subject
import torchvision.transforms as T

from scipy.ndimage import center_of_mass

import nibabel as nib
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image

from diffdrr.drr import DRR, convert
from diffdrr.visualization import plot_img_and_mask, plot_drr, plot_mask
from diffdrr.metrics import NormalizedCrossCorrelation2d
from diffdrr.registration import Registration

from monai.losses.image_dissimilarity import LocalNormalizedCrossCorrelationLoss, GlobalMutualInformationLoss
from monai.losses.multi_scale import MultiScaleLoss
from monai.losses.ssim_loss import SSIMLoss

from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import wandb
from easydict import EasyDict
import yaml
from PIL import Image
from contextlib import nullcontext

from transforms import MedicalImage, ClipROITransform, AABB, ResampleTransform


MM = float | int
Pixel = int
MMPerPixel = float | int

Degree = float | int
DegreePerSec = float | int
Radian = float | int
Angle = TypeVar("Angle", Degree, Radian)
type Rot[Angle] = tuple[Angle, Angle, Angle]

Sec = float | int
Point: tuple[float, float, float]


class LabelID(IntEnum):
    BACKGROUND = 0
    AO = 1
    LV = 2

class WholeHeartLabelID(IntEnum):
    BACKGROUND = 0
    AO = 1
    HEART = 2

@dataclass
class CArmGeometry:
    sdd: MM             # Source to detector distance
    sod: MM             # Source to object distance
    height: Pixel       # Height of image
    delx: MMPerPixel    # Pixel size in x direction
    _width: Pixel | None = None         # Width of image, default to height
    _dely: MMPerPixel | None = None     # Pixel size in y direction, default to delx
    x0: MM = 0.0             # detector principal point x-offset
    y0: MM = 0.0             # detector principal point y-offset
    
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
    

def get_clip_roi_from_label(
    label: Tensor, 
    margin: int = 20,
    keep_xy_shape: bool = True
) -> ClipROITransform:
    """get a ClipROITransform that can crop the volume to the bounding box of the label with a margin
    Args:
        label (Tensor): Binary label mask of shape (1, 1, D, H, W).
        margin (int): Margin to add to the bounding box in voxels.

    Returns:
        ClipROITransform: A transform that can be applied to the volume to crop it to the bounding box of the label with a margin.
    """
    if margin < 0:
        raise ValueError("Margin must be non-negative")
    if label.dim() < 3:
        raise ValueError("Label must have at least 3 dimensions (D, H, W)")
    shape = torch.tensor(label.shape[-3:])  # D, H, W
    
    label = label.squeeze() > 0
    coords = torch.nonzero(label)
    min_coords = coords.min(dim=0).values - margin
    max_coords = coords.max(dim=0).values + margin
    
    min_coords = torch.max(min_coords, torch.zeros_like(min_coords))
    max_coords = torch.min(max_coords, shape - 1)
    
    if keep_xy_shape:
        # keep the original x and y shape, only crop in z direction
        min_coords[:2] = 0
        max_coords[:2] = shape[:2]
    
    return ClipROITransform(AABB(
        x_min=int(min_coords[0]), x_max=int(max_coords[0]),
        y_min=int(min_coords[1]), y_max=int(max_coords[1]),
        z_min=int(min_coords[2]), z_max=int(max_coords[2])
    ))


@dataclass
class DrrSetting:
    patch_size: int|None = None
    orientation_type: Literal["AP", "PA"]|None = "AP"
    parameterization: str = "euler_angles"  # representation of rotation
    convention: str = "ZXY"                 # rotation axis sequence, internal rotation
    mask_to_channels: bool = True           # project masks to different label
    resample_factor: float = 0.5            # downsample volume before DRR rendering


def get_reorientation(
        orientation_type: Optional[Literal["AP", "PA"]] = "AP"
) -> torch.Tensor:
    # Frame-of-reference change
    if orientation_type == "AP":
        # Rotates the C-arm about the x-axis by 90 degrees
        reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    elif orientation_type == "PA":
        # Rotates the C-arm about the x-axis by 90 degrees
        # Reverses the direction of the y-axis
        reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    elif orientation_type is None:
        # Identity transform
        reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unrecognized orientation {orientation_type}")
    return reorient


def recenter(
    original_affine: ndarray,
    center_voxel: tuple[int, ...] | np.ndarray
) -> ndarray:
    """get the affine that set the center of the image to the given center_voxel
    """
    B = original_affine[:3, :3]
    new_t = -(B @ np.array(center_voxel))
    T = np.eye(4)
    T[:3, :3] = B
    T[:3, 3] = new_t
    return T


def get_label_centering_affine(label: Tensor, volume_affine: ndarray) -> ndarray:
    """
    get a new affine that centers the label's structure in world space, by calculating the center of mass of the label and recentering the affine to that point.
    Args:
        label (Tensor): Binary label mask of shape (1, 1, D, H, W).
        volume_affine (ndarray): Original volume affine matrix of shape (4, 4).

    Returns:
        ndarray: New affine matrix that centers the label's structure in world space.
    """
    label_center = center_of_mass((label > 0).squeeze().cpu().numpy()) # type: ignore
    return recenter(volume_affine, label_center)    # type: ignore


def pre_load(file: Path) -> tuple[Nifti1Image, ndarray]:
    image_nii = nib_load(file)
    assert isinstance(image_nii, Nifti1Image)
    affine = image_nii.affine if image_nii.affine is not None else np.eye(4)
    return image_nii, affine


def load_nifti(file: Path, is_label: bool = False) -> MedicalImage  :
    """load nifti file as torch tensor (shape: 1, 1, D, H, W) and return its affine matrix"""
    img, affine = pre_load(file)
    tensor = torch.from_numpy(img.get_fdata())
    if is_label:
        tensor = tensor.round().to(torch.uint8)
    else:
        tensor = tensor.to(torch.float32)
    tensor = tensor[None][None]  # add batch and channel dim
    return MedicalImage(data=tensor, affine=affine)


def build_semantic_masks(label: Tensor, whole_heart_label: Tensor) -> dict[str, Tensor]:
    return {
        "aorta": (label == LabelID.AO) | (whole_heart_label == WholeHeartLabelID.AO),
        "lv": label == LabelID.LV,
        "heart": whole_heart_label == WholeHeartLabelID.HEART,
    }


def adjust_iodine_contrast(
    volume: Tensor, 
    label_to_water: Tensor, 
    label_contrast: Tensor,
    contrast_HU = 200.0
) -> Tensor:
    """
    Adjust the contrast of the iodine-enhanced CTA image based on the label masks.
    Args:
        volume (Tensor): The input CTA volume tensor of shape (1, 1, D, H, W).
        label_to_water (Tensor): The binary label mask tensor of shape (1, 1, D, H, W) > 0 is the to HU = 0 (as water)
        label_un_change (Tensor): The binary label mask tensor of shape (1, 1, D, H, W) > 0 is the area has contrast
    """
    
    res = volume.clone()
    
    # 体外仪器
    res[res > 1500] = res.min()
    
    # 骨骼区域增强
    res[res > 700] *= 1.5
    
    # CTA造影剂区域 与 label_to_warter 区域恢复组织/血液吸收率（与水接近，为0）
    threshold_mask = (volume > 100) & (volume < 400)
    water_mask = (label_to_water > 0) | threshold_mask
    res[water_mask] = 0.
    
    # contrast label 区域增强
    res[label_contrast > 0] += contrast_HU
    
    # HU 值转换为相对值，并clip到合理范围, 组织HU>-200
    res = ((res+200)/1000).clamp(min=0., max=2.)
    
    # normalize to [0, 1]
    res -= res.min()
    res_max = res.max()
    res /= res_max
    
    return res


def get_drr(
    img_path: Path,
    seg_path: Path,
    whole_heart_label_path: Path,
    geom: CArmGeometry,
    device: torch.device,
    config: DrrSetting = DrrSetting()
) -> DRR:
    volume = load_nifti(img_path)
    label = load_nifti(seg_path, is_label=True)
    whole_heart_label = load_nifti(whole_heart_label_path, is_label=True)
    
    volume.to_device(device)
    label.to_device(device)
    whole_heart_label.to_device(device)
    
    print("Clip ROI Original volume shape:", volume.data.shape)
    clip_roi = get_clip_roi_from_label((whole_heart_label.data == WholeHeartLabelID.HEART), margin=40,keep_xy_shape=False)
    volume = clip_roi(volume)
    label = clip_roi(label)
    whole_heart_label = clip_roi(whole_heart_label)
    print("Clip ROI Clipped volume shape:", volume.data.shape)
    
    print(f"Resampling by factor {config.resample_factor} to reduce memory usage and speed up training")
    resample = ResampleTransform(resample_factor=config.resample_factor)
    volume = resample(volume)
    label = resample(label)
    whole_heart_label = resample(whole_heart_label)

    print("Re-centering the volume to the center of mass of the whole heart label")
    affine = get_label_centering_affine((whole_heart_label.data > 0), whole_heart_label.affine)
    masks = build_semantic_masks(label.data, whole_heart_label.data)
    mapped_volume = adjust_iodine_contrast(volume.data, masks["heart"], masks["aorta"])
    
    assert mapped_volume.dim() >= 3
    w, h, d = mapped_volume.shape[-3:]
    shape = (1, w, h, d)
    
    subject = Subject(
        volume=ScalarImage(tensor=mapped_volume.reshape(*shape).float().to(device), affine=affine),
        mask=LabelMap(tensor=label.data.reshape(*shape).float().to(device), affine=affine),
        reorient = get_reorientation("AP"),    # type: ignore
        density = ScalarImage(tensor=mapped_volume.reshape(*shape).float().to(device), affine=affine),
        fiducials = None,   #type: ignore
    )
    
    drr = DRR(
        subject     =   subject,
        sdd         =   geom.sdd,
        height      =   geom.height,
        width       =   geom.width,
        delx        =   geom.delx,
        dely        =   geom.dely,
        x0          =   geom.x0,
        y0          =   geom.y0,
        patch_size  =   config.patch_size,
        checkpoint_gradients=True,
    ).to(device)
    
    return drr


def pose_from_carm(sod: MM, tx: MM, ty: MM, alpha: Radian, beta: Radian, gamma: Radian) -> tuple[Tensor, Tensor]:
    rot = torch.tensor([[alpha, beta, gamma]]).float()
    xyz = torch.tensor([[tx, sod, ty]]).float()
    pose = convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
    rotations, translations = pose.convert("euler_angles", "ZXY")
    return rotations, translations


def project_image(reg: Registration, downsample_stride: int = 1) -> Tensor:
    proj: Tensor = reg(mask_to_channels=True)
    if proj.dim() == 4 and proj.shape[1] > 1:
        img = proj.sum(dim=1, keepdim=True)
    else:
        img = proj if proj.dim() == 4 else proj.unsqueeze(1)
    img = img.max() - img   # reverse as drr projection is density projection, but we need x-ray indensity
    img = img - img.min()
    img = img / img.max().clamp(min=1e-6)
    
    label = (proj > 0).to(torch.uint8).detach().squeeze()[[0, LabelID.AO, LabelID.LV]]
    label[0] = 0
    label = label.argmax(0)
    
    if downsample_stride > 1:
        img = F.avg_pool2d(img, kernel_size=downsample_stride, stride=downsample_stride)
    
    return img

@torch.no_grad()
def valid_project_image_label(reg: Registration) -> tuple[Tensor, Tensor]:
    proj: Tensor = reg(mask_to_channels=True)
    if proj.dim() == 4 and proj.shape[1] > 1:
        img = proj.sum(dim=1, keepdim=True)
    else:
        img = proj if proj.dim() == 4 else proj.unsqueeze(1)
    img = img.max() - img   # reverse as drr projection is density projection, but we need x-ray indensity
    img = img - img.min()
    img = img / img.max().clamp(min=1e-6)
    
    label = (proj > 0).to(torch.uint8).detach().squeeze()[[0, LabelID.AO, LabelID.LV]]
    label[0] = 0
    label = label.argmax(0)

    return img, label

@dataclass
class GridSearchConfig:
    init_value: float
    window_size: float
    step_size: float
    
    def to_candidactes(self) -> list[float]:
        return list(np.arange(
            self.init_value - self.window_size,
            self.init_value + self.window_size + 1e-6,
            self.step_size
        ))

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
    device = gt_img.device if device is None else device

    gt_small = gt_img
    if gt_small.dim() == 3:
        gt_small = gt_small.unsqueeze(0)
    if img_downsample_stride > 1:
        gt_small = F.avg_pool2d(gt_small, kernel_size=img_downsample_stride, stride=img_downsample_stride)

    alpha_candidates = alpha_grid_config.to_candidactes()
    beta_candidates = beta_grid_config.to_candidactes()
    sod_candidates = sod_grid_config.to_candidactes()

    best_score = float("inf")
    best_alpha_deg: float = 0.0
    best_beta_deg: float = 0.0
    best_sod_mm: float = 0.0
    best_rotations: Tensor | None = None
    best_translations: Tensor | None = None
    
    pbar = tqdm(
        product(alpha_candidates, beta_candidates, sod_candidates), 
        total=len(alpha_candidates)*len(beta_candidates)*len(sod_candidates), 
        desc="Coarse grid search", 
        ncols=120
    )
    for alpha_deg, beta_deg, sod_mm in pbar:
        rotations, translations = pose_from_carm(
            sod_mm,
            0.0,
            0.0,
            radians(alpha_deg),
            radians(beta_deg),
            radians(init_gamma_deg),   # use alpha as gamma init since they are both rotation about x-axis
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
        pbar.set_postfix(best_score=f"{best_score:.6f}", alpha=f"{best_alpha_deg:.2f}", beta=f"{best_beta_deg:.2f}", sod=f"{best_sod_mm:.2f}")
    
    
    assert best_rotations is not None and best_translations is not None
    return best_rotations, best_translations


def run(reg: Registration, downsample_stride: int = 1) -> tuple[Tensor, Tensor]:
    img = project_image(reg, downsample_stride=downsample_stride)
    
    proj: Tensor = reg(mask_to_channels=True)
    label = (proj > 0).to(torch.uint8).detach().squeeze()[[0, LabelID.AO, LabelID.LV]]
    label[0] = 0
    label = label.argmax(0)
    return img, label


def run_image_only(reg: Registration, downsample_stride: int = 1) -> Tensor:
    return project_image(reg, downsample_stride=downsample_stride)


@torch.no_grad()
def run_label_only(reg: Registration) -> Tensor:
    proj: Tensor = reg(mask_to_channels=True)
    label = (proj > 0).to(torch.uint8).detach().squeeze()[[0, LabelID.AO, LabelID.LV]]
    label[0] = 0
    return label.argmax(0)


def _sobel_filters(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
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
        pred_centered.square().mean(dim=(-2, -1)) * target_centered.square().mean(dim=(-2, -1)) + 1e-8
    )
    return numerator / denominator


class ValidMetricKeys(StrEnum):
    MSE = "val/mse"
    MAE = "val/mae"
    NCC = "val/ncc"
    PSNR = "val/psnr"
    GRAD_MSE = "val/grad_mse"
    SSIM = "val/ssim"
    MI = "val/mi"



class CheckpointedLNCC(nn.Module):
    def __init__(self, lncc: LocalNormalizedCrossCorrelationLoss):
        super().__init__()
        self.lncc = lncc

    def forward(self, pred, target):
        return  torch.utils.checkpoint.checkpoint(self.lncc, pred, target)  #type: ignore

@dataclass
class StageConfig:
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
    def from_dict(d: dict[str, Any]) -> 'StageConfig':
        return StageConfig(
            name=d.get("name", "stage"),
            downsample_stride=int(d.get("downsample_stride", 1)),
            n_itrs=int(d.get("n_itrs", 0)),
            lr_trans_mult=float(d.get("lr_trans_mult", 1.0)),
        )


def validation_score(metrics: dict[ValidMetricKeys, float]) -> float:
    return (
        1.0 * metrics[ValidMetricKeys.NCC]
        + 1.0 * metrics[ValidMetricKeys.PSNR]
        - 1.0 * metrics[ValidMetricKeys.MSE]
        - 1.0 * metrics[ValidMetricKeys.MAE]
        - 1.0 * metrics[ValidMetricKeys.GRAD_MSE]
        + 5.0 * metrics[ValidMetricKeys.SSIM]
        + 5.0 * metrics[ValidMetricKeys.MI]
    )

class BestMetricTracker:
    def __init__(self):
        self.best_score = float("-inf")
        self.best_metrics: dict[ValidMetricKeys, float] | None = None
        self.best_rotateions: Tensor | None = None
        self.best_translations: Tensor | None = None
        self.best_reg_state_dict: dict[str, Tensor] | None = None

    def maybe_store(self, reg: Registration, metrics: dict[ValidMetricKeys, float]) -> None:
        score = validation_score(metrics)
        if score > self.best_score:
            self.best_score = score
            self.best_metrics = metrics
            self.best_rotateions = reg.rotation.clone().detach()
            self.best_translations = reg.translation.clone().detach()
            self.best_reg_state_dict = reg.state_dict()


class Validation:
    def __init__(
        self,
        stage_name: str, 
        reg: Registration,
        gt_img: Tensor, 
        loss_fn: Callable, 
        best_metric_tracker: BestMetricTracker
    ):
        self.stage_name = stage_name
        self.reg = reg
        self.gt_img = gt_img
        self.loss_fn = loss_fn
        self.best_metric_tracker = best_metric_tracker
    
    @property
    def best_metrics(self) -> dict[ValidMetricKeys, float]:
        assert self.best_metric_tracker.best_metrics is not None, "Best metrics should not be None since we validated at least once before training"
        return self.best_metric_tracker.best_metrics
    
    @property
    def best_pose(self) -> tuple[Tensor, Tensor]:
        assert self.best_metric_tracker.best_rotateions is not None and self.best_metric_tracker.best_translations is not None, "Best rotations and translations should not be None since we validated at least once before training"
        return self.best_metric_tracker.best_rotateions, self.best_metric_tracker.best_translations
    
    @property
    def best_reg_state_dict(self) -> dict[str, Tensor]:
        assert self.best_metric_tracker.best_reg_state_dict is not None, "Best reg state dict should not be None since we validated at least once before training"
        return self.best_metric_tracker.best_reg_state_dict
    
    def validate_predictions(self, pred_img: Tensor) -> dict[ValidMetricKeys, float]:
        gt_img = self.gt_img.detach()
        pred_img = pred_img.detach()
        
        if gt_img.dim() == 3:
            gt_img = gt_img.unsqueeze(0)
        if pred_img.dim() == 3:
            pred_img = pred_img.unsqueeze(0)

        gt_img = gt_img.to(torch.float32)
        pred_img = pred_img.to(torch.float32)

        mse = F.mse_loss(pred_img, gt_img)
        mae = F.l1_loss(pred_img, gt_img)
        ncc = _ncc_2d(pred_img, gt_img).mean()
        psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-8))
        grad_mse = F.mse_loss(_gradient_magnitude(pred_img), _gradient_magnitude(gt_img))
        
        ssim_loss = SSIMLoss(spatial_dims=2)
        ssim = 1.0 - ssim_loss(pred_img, gt_img)
        
        mi_loss = GlobalMutualInformationLoss()
        mi = -mi_loss(pred_img, gt_img) 

        return {
            ValidMetricKeys.MSE: float(mse.item()),
            ValidMetricKeys.MAE: float(mae.item()),
            ValidMetricKeys.NCC: float(ncc.item()),
            ValidMetricKeys.PSNR: float(psnr.item()),
            ValidMetricKeys.GRAD_MSE: float(grad_mse.item()),
            ValidMetricKeys.SSIM: float(ssim.item()),
            ValidMetricKeys.MI: float(mi.item()),
        }
    
    
    @torch.no_grad()
    def __call__(self, iter: int) -> None:
        img, label = valid_project_image_label(self.reg)
        val_metrics = self.validate_predictions(img)
        val_score = validation_score(val_metrics)
        loss: Tensor = self.loss_fn(self.gt_img, img)
        
        masks={
            "labels": {
                'mask_data': label,
                "class_labels": {
                    1: "AO",
                    2: "LV"
                }
            }
        }
        
        wandb.log({**val_metrics, "val/score": val_score, "val/loss": float(loss.detach().item()), "stage": self.stage_name, "iter": iter})
        self.best_metric_tracker.maybe_store(self.reg, val_metrics)
        wandb.log(
            {
                "drr_image": wandb.Image(img.detach().squeeze()[None], mode="L", masks=masks),
                "gt_image": wandb.Image(self.gt_img.squeeze()[None], mode="L", masks=masks),
            }
        )


def train(
    reg: Registration,
    gt_img_resampled: Tensor,
    gt_img: Tensor,
    optim: torch.optim.Optimizer,
    loss_fn: Callable,
    n_itrs: int,
    val_intervals: int,
    downsample_stride: int = 1,
    stage_name: str = "fine"
) -> tuple[dict[ValidMetricKeys, float], Tensor, Tensor]:
    # Initialize an optimizer with different learning rates
    # for rotations and translations since they have different scales
    wandb.watch(reg.drr)
    amp_enabled = False
    
    # record init value
    def get_record(iter_: int, reg_: Registration, loss_: Tensor) -> dict[str, int|float|str]:
        alpha, beta, gamma = reg_.rotation.squeeze().tolist()
        bx, by, bz = reg_.translation.squeeze().tolist()
        return {"loss": float(loss_.item()),
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "bx": bx, "by": by, "bz": bz,
            "stage": stage_name,
        }

    valid_fn = Validation(stage_name, reg, gt_img, loss_fn, BestMetricTracker())
    
    valid_fn(0)   # validate before training
    for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
        optim.zero_grad(set_to_none=True)
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                img = run_image_only(reg, downsample_stride=downsample_stride)
                loss: Tensor = loss_fn(gt_img_resampled, img)
        else:
            img = run_image_only(reg, downsample_stride=downsample_stride)
            loss = loss_fn(gt_img_resampled, img)

        loss.backward()
        optim.step()
        
        # Log current learning rates
        current_lr_rot = optim.param_groups[0]['lr']
        current_lr_trans = optim.param_groups[1]['lr']
        
        pbar.set_description(f"[{stage_name}] loss = {loss.item():06f}")
        record = get_record(itr, reg, loss.detach())
        record["lr_rotations"] = current_lr_rot
        record["lr_translations"] = current_lr_trans
        wandb.log(record)
        
        if itr >= val_intervals and itr % val_intervals == 0:
            valid_fn(itr)

    best_score = valid_fn.best_metric_tracker.best_score
    best_rotations, best_translations = valid_fn.best_pose
    final_metrics = valid_fn.best_metrics
    # best_state_dict = valid_fn.best_reg_state_dict
    # reg.load_state_dict(best_state_dict)

    print(f"[stage-summary] {stage_name}: best_score={best_score:.6f}, best_mse={final_metrics.get(ValidMetricKeys.MSE, float('nan')):.6f}, best_ncc={final_metrics.get(ValidMetricKeys.NCC, float('nan')):.6f}")
    wandb.log({
        "stage": stage_name,
        "stage/best_score": best_score,
        "stage/best_mse": final_metrics.get(ValidMetricKeys.MSE, float("nan")),
        "stage/best_ncc": final_metrics.get(ValidMetricKeys.NCC, float("nan")),
    })
    
    return final_metrics, best_rotations, best_translations


def load_config(path: Path) -> EasyDict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def get_loss_fn(cfg: dict[str, Any]|list[dict[str, Any]], device: torch.device) -> Callable:
    def type_to_loss_fn_map(type_str: str, init_args: dict[str, Any]) -> Callable:
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

        loss_fns.append((weight, type_to_loss_fn_map(loss_type, init_args)))
    
    def res(x, y):
        total_loss = 0.0
        for weight, loss_fn in loss_fns:
            total_loss += weight * loss_fn(x, y)
        return total_loss
    
    return res
    
    
def main():
    config_path = Path("config/base_config.yaml")
    cfg: Any = load_config(config_path)

    # load config and init wandb
    wandb.init(
        project="aortic_regurgitation",
        config=cfg,
    )

    # init DRR with geometry
    geom = CArmGeometry(
        sdd     =   cfg.geom.sdd,
        sod     =   cfg.geom.sod,
        height  =   cfg.geom.height,
        delx    =   cfg.geom.delx
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    drr = get_drr(
        Path(cfg.data.img_path),
        Path(cfg.data.seg_path),
        Path(cfg.data.whole_heart_label_path),
        geom, 
        device,
        config=DrrSetting(**cfg.get("drr", {})),
    ).eval()   # set DRR to eval mode since we don't need autograd for the DRR itself, only for the registration parameters

    img_2d = T.ToTensor()(
        T.Resize((cfg.geom.height, cfg.geom.height))(
            Image.open(cfg.data.image_2d_path).convert("L")
        )
    )[None].to(device)

    loss_fn = get_loss_fn(cfg.loss, device)
    
    # init pose: first do a cheap coarse search over the two main C-arm angles
    # rotations, translations = coarse_init_pose(
    #     drr,
    #     img_2d,
    #     alpha_grid_config=GridSearchConfig(init_value=cfg.init_pose.alpha, **cfg.coarse_init.alpha_grid),
    #     beta_grid_config=GridSearchConfig(init_value=cfg.init_pose.beta, **cfg.coarse_init.beta_grid),
    #     sod_grid_config=GridSearchConfig(geom.sod, **cfg.coarse_init.sod_grid),
    #     loss_fn=loss_fn,
    #     img_downsample_stride=cfg.coarse_init.img_downsample_stride,
    #     init_gamma_deg=cfg.coarse_init.init_gamma_deg,
    #     device=device,
    # )
    
    rotations, translations = pose_from_carm(
        geom.sod, 0, 0,
        radians(cfg.init_pose.alpha), radians(cfg.init_pose.beta), 0.
    )

    # init registration for training
    reg = Registration(
        drr,
        rotations.to(device),
        translations.to(device),
        parameterization="euler_angles",
        convention="ZXY",
    )

    if cfg.optimizer.type.upper() == "SGD":
        optim_cls = torch.optim.SGD
    elif cfg.optimizer.type.upper() == "ADAM":
        optim_cls = torch.optim.Adam
    else:
        raise NotImplementedError

 
    optim = optim_cls(
        [
            {"params": [reg._rotation], "lr": cfg.optimizer.lr_rotations},
            {"params": [reg._translation], "lr": cfg.optimizer.lr_translations},
        ],
        **cfg.optimizer.init_args
    )
    
    coarse_cfg = cfg.get("coarse_to_fine", None)
    if coarse_cfg is not None and coarse_cfg.get("stages", None):
        stages = [StageConfig.from_dict(stage) for stage in coarse_cfg.stages]
    else:
        total_itrs = int(cfg.train.n_itrs)
        coarse_itrs = max(1, total_itrs // 3)
        mid_itrs = max(1, total_itrs // 3)
        fine_itrs = max(1, total_itrs - coarse_itrs - mid_itrs)
        stages = [
            StageConfig(name="coarse", downsample_stride=4, n_itrs=coarse_itrs, lr_rot_mult=0.25, lr_trans_mult=0.25),
            StageConfig(name="mid", downsample_stride=2, n_itrs=mid_itrs, lr_rot_mult=0.5, lr_trans_mult=0.5),
            StageConfig(name="fine", downsample_stride=1, n_itrs=fine_itrs, lr_rot_mult=1.0, lr_trans_mult=1.0),
        ]

    stage_summaries: list[dict[str, Any]] = []
    for stage in stages:
        if int(stage.downsample_stride) > 1:
            gt_img_resampled  =   F.avg_pool2d(
                img_2d, 
                kernel_size=stage.downsample_stride, 
                stride=stage.downsample_stride
            ) 
        else:
            gt_img_resampled = img_2d
        
        best_metrics, _, _ = train(
            reg     =   reg,
            gt_img_resampled  =   gt_img_resampled,
            gt_img  =   img_2d,
            optim   =   optim,
            loss_fn =   loss_fn,
            n_itrs  =   stage.n_itrs,
            val_intervals = cfg.train.val_interval,
            downsample_stride = stage.downsample_stride,
            stage_name = stage.name,
        )
        stage_summaries.append({
            "stage": stage.name,
            "best_score": validation_score(best_metrics) if best_metrics else float("nan"),
            **best_metrics,
        })

    print("[run-summary] stage best metrics:")
    for summary in stage_summaries:
        print(summary)


if __name__ == "__main__":
    main()