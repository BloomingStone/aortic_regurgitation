from typing import Literal, Optional, TypeVar, Any, Callable
from dataclasses import dataclass
from math import radians
from pathlib import Path
from enum import IntEnum

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
from torch import nn
from torch.utils.dlpack import from_dlpack as dlpack2tensor
from torch.utils.dlpack import to_dlpack as tensor2dlpack   #type: ignore
from torchio import LabelMap, ScalarImage, Subject
import torchvision.transforms as T

import cupy as cp
from cupyx.scipy.ndimage import center_of_mass, binary_dilation, binary_closing

import nibabel as nib
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image

from diffdrr.drr import DRR, convert
from diffdrr.visualization import plot_img_and_mask, plot_drr, plot_mask
from diffdrr.metrics import NormalizedCrossCorrelation2d
from diffdrr.registration import Registration

from monai.losses.image_dissimilarity import LocalNormalizedCrossCorrelationLoss, GlobalMutualInformationLoss
from monai.losses.multi_scale import MultiScaleLoss

from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import wandb
from easydict import EasyDict
import yaml
from PIL import Image


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
    AO = 1
    LV = 2
    # AO = 1
    # LA = 2
    # RA = 3
    # LVM = 4
    # LV = 5
    # RV = 6
    # PV = 7

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


@dataclass
class DrrSetting:
    patch_size: int|None = None
    orientation_type: Literal["AP", "PA"]|None = "AP"
    parameterization: str = "euler_angles"  # representation of rotation
    convention: str = "ZXY"                 # rotation axis sequence, internal rotation
    mask_to_channels: bool = True           # project masks to different label


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
    center_voxel: tuple[int, int, int]
) -> ndarray:
    """get the affine that set the center of the image to the given center_voxel
    """
    B = original_affine[:3, :3]
    new_t = -(B @ np.array(center_voxel))
    T = np.eye(4)
    T[:3, :3] = B
    T[:3, 3] = new_t
    return T


def get_coronary_centering_affine(label: Tensor, volume_affine: ndarray) -> ndarray:
    """
    Compute an affine transform that recenters the heart region in world coordinates for CTA
    Args:
        label (Tensor): Binary label mask of shape (1, 1, D, H, W).
        volume_affine (ndarray): Original volume affine matrix of shape (4, 4).

    Returns:
        ndarray: New affine matrix that centers the label's structure in world space.
    """
    ref_label = (label > 0) & (label!=LabelID.AO)
    with cp.cuda.Device(label.device.index):
        label_center = center_of_mass(cp.from_dlpack(tensor2dlpack(ref_label.squeeze().to(label.device)))) # type: ignore
        label_center = (int(label_center[0]), int(label_center[1]), int(label_center[2]))
        return recenter(volume_affine, label_center)


def pre_load(file: Path) -> tuple[Nifti1Image, ndarray]:
    image_nii = nib_load(file)
    assert isinstance(image_nii, Nifti1Image)
    affine = image_nii.affine if image_nii.affine is not None else np.eye(4)
    return image_nii, affine


def load_nifti(file: Path, is_label: bool = False) -> tuple[Tensor, ndarray]:
    """load nifti file as torch tensor (shape: 1, 1, D, H, W) and return its affine matrix"""
    img, affine = pre_load(file)
    tensor = torch.from_numpy(img.get_fdata())
    if is_label:
        tensor = tensor.round().to(torch.uint8)
    else:
        tensor = tensor.to(torch.float32)
    tensor = tensor[None][None]  # add batch and channel dim
    return tensor, affine


def adjust_iodine_contrast(volume: Tensor, label: Tensor) -> Tensor:
    res = volume.clone()
    # CTA造影剂区域恢复正常血液（与水接近，为0）
    threshold_mask = (volume > 150) & (volume < 400)
    res[(label > 0) | threshold_mask ] = 0.
    
    # 体外仪器的线
    res[res > 1500] = res.min()
    res[res > 700] *= 1.2
    res[(res > -150) & (res < 0.)] = 0.
    res[res < -150] = res.min()
    
    res = (res/1000 + 1).clamp(min=0., max=2.)
    res_max = res.max()
    res /= res_max
    res = res ** 2
    res *= res_max
    return res


def get_drr(
    img_path: Path,
    seg_path: Path,
    whole_heart_label_path: Path,
    geom: CArmGeometry,
    device: torch.device,
    config: DrrSetting = DrrSetting()
) -> DRR:
    volume, volume_affine = load_nifti(img_path)
    label, _ = load_nifti(seg_path, is_label=True)
    whole_heart_label, _ = load_nifti(whole_heart_label_path, is_label=True)
    volume = volume.to(device)
    label = label.to(device)
    whole_heart_label = whole_heart_label.to(device)
    
    bone_mask = (volume > 700).squeeze()
    bone_mask_cp = cp.from_dlpack(tensor2dlpack(bone_mask.to(device)))
    bone_mask_cp = binary_closing(bone_mask_cp)
    bone_mask = dlpack2tensor(bone_mask_cp.toDlpack())
    bone_mask_coords = torch.where(bone_mask)
    w, h, d = volume.shape[-3:]
    x_min = bone_mask_coords[0].min()
    x_max = bone_mask_coords[0].max()
    y_min = bone_mask_coords[1].min()
    y_max = bone_mask_coords[1].max()
    z_min = bone_mask_coords[2].min()
    z_max = bone_mask_coords[2].max()
    
    volume = volume[:, :, x_min:x_max, y_min:y_max, z_min:z_max]
    label = label[:, :, x_min:x_max, y_min:y_max, z_min:z_max]
    whole_heart_label = whole_heart_label[:, :, x_min:x_max, y_min:y_max, z_min:z_max]
    

    whole_heart_label[whole_heart_label != 2] = 0
    affine = get_coronary_centering_affine(whole_heart_label, volume_affine)
    volume = adjust_iodine_contrast(volume, label)
    
    assert volume.dim() >= 3
    w, h, d = volume.shape[-3:]
    shape = (1, w, h, d)
    
    subject = Subject(
        volume=ScalarImage(tensor=volume.reshape(*shape).to(device), affine=affine),
        mask=LabelMap(tensor=label.reshape(*shape).to(device), affine=affine),
        reorient = get_reorientation("AP"),    # type: ignore
        density = ScalarImage(tensor=volume.reshape(*shape).to(device), affine=affine),
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
        renderer    =   "trilinear",
        checkpoint_gradients=True
    ).to(device)
    
    return drr


def pose_from_carm(sod: MM, tx: MM, ty: MM, alpha: Radian, beta: Radian, gamma: Radian) -> tuple[Tensor, Tensor]:
    rot = torch.tensor([[alpha, beta, gamma]])
    xyz = torch.tensor([[tx, sod, ty]])
    pose = convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
    rotations, translations = pose.convert("euler_angles", "ZXY")
    return rotations, translations


def run(reg: Registration) -> tuple[Tensor, Tensor]:
    proj: Tensor = reg(mask_to_channels=True)
    img = proj.sum(dim=1, keepdim=True)
    img = img.max() - img
    img = img / img.max()

    label = (proj > 0).to(torch.uint8).detach().squeeze()[[0, LabelID.AO, LabelID.LV]]
    del proj
    label[0] = 0
    label = label.argmax(0)
    return img, label


def run_with_masks(reg: Registration) -> tuple[Tensor, Tensor, Tensor]:
    """返回 img, combined_label, individual_masks
    individual_masks: shape (2, H, W), 分别对应 AO 和 LV 的 mask
    """
    proj: Tensor = reg(mask_to_channels=True)
    img = proj.sum(dim=1, keepdim=True)
    img = img.max() - img
    img = img / img.max()

    # 分别获取 AO 和 LV 的 mask
    # proj 形状: (num_labels, H, W), 索引 0 是背景, 1 是 AO, 2 是 LV
    masks = (proj > 0).to(torch.float32).detach().squeeze()  # (3, H, W)
    ao_mask = masks[LabelID.AO]  # AO
    lv_mask = masks[LabelID.LV]  # LV

    del proj
    return img, ao_mask, lv_mask


def save_label_overlay(
    gt_img: Tensor,
    ao_mask: Tensor,
    lv_mask: Tensor,
    output_path: Path,
    alpha: float = 0.7
):
    """将 label 叠加到 gt_image 上保存
    AO: 红色, LV: 绿色
    alpha: mask 透明度 (0-1)，默认为 0.7 (70%)
    """
    # 转换为 numpy 并归一化到 0-255
    gt_np = (gt_img.squeeze().cpu().detach() * 255).to(torch.uint8).numpy()
    ao_np = (ao_mask.squeeze().cpu().detach().numpy() > 0).astype(np.uint8) * 255
    lv_np = (lv_mask.squeeze().cpu().detach().numpy() > 0).astype(np.uint8) * 255

    # 创建 RGB 图像
    rgb = np.stack([gt_np, gt_np, gt_np], axis=-1).astype(np.float32)

    # 叠加 AO (红色) - 带透明度
    ao_mask_bool = ao_np > 0
    rgb[ao_mask_bool, 0] = rgb[ao_mask_bool, 0] * (1 - alpha) + 255 * alpha
    rgb[ao_mask_bool, 1] = rgb[ao_mask_bool, 1] * (1 - alpha) + 0 * alpha
    rgb[ao_mask_bool, 2] = rgb[ao_mask_bool, 2] * (1 - alpha) + 0 * alpha

    # 叠加 LV (绿色) - 带透明度
    lv_mask_bool = lv_np > 0
    rgb[lv_mask_bool, 0] = rgb[lv_mask_bool, 0] * (1 - alpha) + 0 * alpha
    rgb[lv_mask_bool, 1] = rgb[lv_mask_bool, 1] * (1 - alpha) + 255 * alpha
    rgb[lv_mask_bool, 2] = rgb[lv_mask_bool, 2] * (1 - alpha) + 0 * alpha

    # 保存
    Image.fromarray(rgb.astype(np.uint8)).save(output_path)


def save_masks_only(
    ao_mask: Tensor,
    lv_mask: Tensor,
    output_path: Path
):
    """保存单独的 mask 图片
    AO: 红色, LV: 绿色, 背景: 黑色
    """
    # 转换为 numpy
    ao_np = (ao_mask.squeeze().cpu().detach().numpy() > 0).astype(np.uint8)
    lv_np = (lv_mask.squeeze().cpu().detach().numpy() > 0).astype(np.uint8)

    # 创建 RGB 图像 (黑色背景)
    h, w = ao_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # AO 红色
    rgb[ao_np > 0, 0] = 255

    # LV 绿色
    rgb[lv_np > 0, 1] = 255

    # 保存
    Image.fromarray(rgb).save(output_path)

class CheckpointedLNCC(nn.Module):
    def __init__(self, lncc: LocalNormalizedCrossCorrelationLoss):
        super().__init__()
        self.lncc = lncc

    def forward(self, pred, target):
        return  torch.utils.checkpoint.checkpoint(self.lncc, pred, target)  #type: ignore

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: dict
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Create learning rate scheduler from configuration."""
    if scheduler_cfg is None:
        return None

    scheduler_type = scheduler_cfg.get("type", "").upper()

    if scheduler_type == "STEP":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 50),
            gamma=scheduler_cfg.get("gamma", 0.5)
        )
    elif scheduler_type == "EXPONENTIAL":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_cfg.get("gamma", 0.95)
        )
    elif scheduler_type == "COSINE":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", 100),
            eta_min=scheduler_cfg.get("eta_min", 0)
        )
    elif scheduler_type == "REDUCEONPLATEAU":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 10),
            threshold=scheduler_cfg.get("threshold", 1e-4)
        )
    elif scheduler_type == "ONECYCLE":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_cfg.get("max_lr", 0.1),
            total_steps=scheduler_cfg.get("total_steps", 100),
            pct_start=scheduler_cfg.get("pct_start", 0.3)
        )
    elif scheduler_type == "POLY":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / scheduler_cfg.get("T_max", 100)) ** scheduler_cfg.get("power", 0.9)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train(
    reg: Registration,
    gt_img: Tensor,
    optim: torch.optim.Optimizer,
    loss_fn: Callable,
    n_itrs: int,
    val_intervals: int,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    save_dir: Optional[Path] = None
):
    # Initialize an optimizer with different learning rates
    # for rotations and translations since they have different scales
    wandb.watch(reg.drr)

    # Create save directory
    if save_dir is None:
        save_dir = Path("results")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # record init value
    def get_record(iter_: int, reg_: Registration, loss_: Tensor) -> dict[str, int|float]:
        alpha, beta, gamma = reg_.rotation.squeeze().tolist()
        bx, by, bz = reg_.translation.squeeze().tolist()
        return {"loss": float(loss_.item()),
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "bx": bx, "by": by, "bz": bz
        }

    img, label = run(reg)
    record = get_record(0, reg, loss_fn(gt_img, img))

    # Save initial result (第1次)
    img, ao_mask, lv_mask = run_with_masks(reg)
    save_label_overlay(gt_img, ao_mask, lv_mask, save_dir / "01_init_gt_mask.png", alpha=0.7)
    save_masks_only(ao_mask, lv_mask, save_dir / "01_init_mask.png")
    Image.fromarray((img.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)).save(save_dir / "01_init_drr.png")

    # 初始保存点索引为0，用于生成后续索引
    save_point_indices = {n_itrs * i // 10: i + 2 for i in range(1, 10)}  # {10%:2, 20%:3, ..., 90%:10}

    # 计算保存点：共10次 (0%, 10%, 20%, ..., 90%, 100%)
    # save_point_indices: {迭代次数: 保存索引}，如 {10%:2, 20%:3, ..., 90%:10}
    save_point_indices = {n_itrs * i // 10: i + 2 for i in range(1, 10)}

    for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
        optim.zero_grad()
        img, label = run(reg)

        loss: Tensor = loss_fn(gt_img, img)
        loss.backward()
        optim.step()

        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.item())
            else:
                scheduler.step()

        # Log current learning rates
        current_lr_rot = optim.param_groups[0]['lr']
        current_lr_trans = optim.param_groups[1]['lr']

        pbar.set_description(f"loss = {loss.item():06f}")
        record = get_record(itr, reg, loss.detach())
        record["lr_rotations"] = current_lr_rot
        record["lr_translations"] = current_lr_trans
        wandb.log(record)

        # Save at specific intervals (第2-9次)
        if itr in save_point_indices:
            save_idx = save_point_indices[itr]  # 2, 3, ..., 10
            img_val, ao_mask, lv_mask = run_with_masks(reg)

            masks={
                "labels": {
                    'mask_data': label,
                    "class_labels": {
                        1: "AO",
                        2: "LV"
                    }
                }
            }
            wandb.log(
                {
                    "drr_image": wandb.Image(img_val.detach().squeeze()[None], mode="L", masks=masks),
                    "gt_image": wandb.Image(gt_img.squeeze()[None], mode="L", masks=masks),
                }
            )

            # 保存3张图
            prefix = f"{save_idx:02d}_iter{itr:04d}"
            save_label_overlay(gt_img, ao_mask, lv_mask, save_dir / f"{prefix}_gt_mask.png", alpha=0.7)
            save_masks_only(ao_mask, lv_mask, save_dir / f"{prefix}_mask.png")
            Image.fromarray((img_val.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)).save(save_dir / f"{prefix}_drr.png")

    # Save final result (第10次)
    img_final, ao_mask_final, lv_mask_final = run_with_masks(reg)
    save_label_overlay(gt_img, ao_mask_final, lv_mask_final, save_dir / "10_final_gt_mask.png", alpha=0.7)
    save_masks_only(ao_mask_final, lv_mask_final, save_dir / "10_final_mask.png")
    Image.fromarray((img_final.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)).save(save_dir / "10_final_drr.png")
    print(f"Results saved to {save_dir}")


def load_config(path: Path) -> EasyDict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


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
        device
    ).train()

    # init pose
    rotations, translations = pose_from_carm(
        geom.sod, 0., 0., 
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

    if cfg.loss.type.upper() == "NCC":
        ncc = NormalizedCrossCorrelation2d().to(device)
        loss_fn = lambda x, y: ncc(x, y)
    elif cfg.loss.type.upper() == "LNCC":
        lncc = LocalNormalizedCrossCorrelationLoss(**cfg.loss.init_args).to(device)
        loss_fn = CheckpointedLNCC(lncc)
    elif cfg.loss.type.upper() == "MI":
        loss_fn = GlobalMutualInformationLoss(**cfg.loss.init_args).to(device)
    else:
        raise NotImplementedError
    
    if "multiscale" in cfg.loss and cfg.loss.multiscale is not None:
        loss_fn = MultiScaleLoss(loss_fn, scales = cfg.loss.multiscale)    # type: ignore

    # Initialize learning rate scheduler
    scheduler = get_scheduler(optim, cfg.get("scheduler", None))


    img_2d = T.ToTensor()(
        Image.open(cfg.data.image_2d_path).convert("L")
    )[None].to(device)

    train(
        reg     =   reg,
        gt_img  =   img_2d,
        optim   =   optim,
        loss_fn =   loss_fn,
        n_itrs  =   cfg.train.n_itrs,
        val_intervals = cfg.train.val_interval,
        scheduler = scheduler,
        save_dir = Path("results")
    )


if __name__ == "__main__":
    main()