"""训练循环、验证、指标追踪与学习率调度器。"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import lr_scheduler

from tqdm import tqdm
import wandb

from diffdrr.registration import Registration

from monai.losses.image_dissimilarity import GlobalMutualInformationLoss
from monai.losses.ssim_loss import SSIMLoss

from common_types import ValidMetricKeys
from drr import run_image_only, valid_project_image_label, run_with_masks
from losses import (
    _gradient_magnitude,
    _ncc_2d,
    validation_score,
)

# ── 学习率调度器工厂 ────────────────────────────────────────────

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: dict | None,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """从配置创建学习率调度器。

    支持类型：STEP, EXPONENTIAL, COSINE, REDUCEONPLATEAU, ONECYCLE, POLY。
    """
    if scheduler_cfg is None:
        return None

    scheduler_type = scheduler_cfg.get("type", "").upper()

    if scheduler_type == "STEP":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 50),
            gamma=scheduler_cfg.get("gamma", 0.5),
        )
    elif scheduler_type == "EXPONENTIAL":
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_cfg.get("gamma", 0.95),
        )
    elif scheduler_type == "COSINE":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", 100),
            eta_min=scheduler_cfg.get("eta_min", 0),
        )
    elif scheduler_type == "REDUCEONPLATEAU":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 10),
            threshold=scheduler_cfg.get("threshold", 1e-4),
        )
    elif scheduler_type == "ONECYCLE":
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_cfg.get("max_lr", 0.1),
            total_steps=scheduler_cfg.get("total_steps", 100),
            pct_start=scheduler_cfg.get("pct_start", 0.3),
        )
    elif scheduler_type == "POLY":
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / scheduler_cfg.get("T_max", 100))
            ** scheduler_cfg.get("power", 0.9),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ── 指标追踪器 ──────────────────────────────────────────────────

class BestMetricTracker:
    """追踪最佳验证指标及对应的位姿。"""

    def __init__(self):
        self.best_score: float = float("-inf")
        self.best_metrics: dict[ValidMetricKeys, float] | None = None
        self.best_rotations: Tensor | None = None
        self.best_translations: Tensor | None = None
        self.best_reg_state_dict: dict[str, Tensor] | None = None

    def maybe_store(
        self, reg: Registration, metrics: dict[ValidMetricKeys, float]
    ) -> None:
        score = validation_score(metrics)
        if score > self.best_score:
            self.best_score = score
            self.best_metrics = metrics
            self.best_rotations = reg.rotation.clone().detach()
            self.best_translations = reg.translation.clone().detach()
            self.best_reg_state_dict = reg.state_dict()


# ── 验证 ────────────────────────────────────────────────────────

class Validation:
    """单次验证：计算多指标并记录到 wandb。"""

    def __init__(
        self,
        stage_name: str,
        reg: Registration,
        gt_img: Tensor,
        loss_fn: Callable,
        best_metric_tracker: BestMetricTracker,
    ):
        self.stage_name = stage_name
        self.reg = reg
        self.gt_img = gt_img
        self.loss_fn = loss_fn
        self.best_metric_tracker = best_metric_tracker

    @property
    def best_metrics(self) -> dict[ValidMetricKeys, float]:
        assert self.best_metric_tracker.best_metrics is not None
        return self.best_metric_tracker.best_metrics

    @property
    def best_pose(self) -> tuple[Tensor, Tensor]:
        assert self.best_metric_tracker.best_rotations is not None
        assert self.best_metric_tracker.best_translations is not None
        return (
            self.best_metric_tracker.best_rotations,
            self.best_metric_tracker.best_translations,
        )

    def validate_predictions(self, pred_img: Tensor) -> dict[ValidMetricKeys, float]:
        gt = self.gt_img.detach().to(torch.float32)
        pred = pred_img.detach().to(torch.float32)

        if gt.dim() == 3:
            gt = gt.unsqueeze(0)
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)

        mse = F.mse_loss(pred, gt)
        mae = F.l1_loss(pred, gt)
        ncc = _ncc_2d(pred, gt).mean()
        psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-8))
        grad_mse = F.mse_loss(
            _gradient_magnitude(pred), _gradient_magnitude(gt)
        )

        ssim_loss = SSIMLoss(spatial_dims=2)
        ssim = 1.0 - ssim_loss(pred, gt)

        mi_loss = GlobalMutualInformationLoss()
        mi = -mi_loss(pred, gt)

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

        masks = {
            "labels": {
                "mask_data": label,
                "class_labels": {1: "AO", 2: "LV"},
            }
        }

        wandb.log({
            **val_metrics,
            "val/score": val_score,
            "val/loss": float(loss.detach().item()),
            "stage": self.stage_name,
            "iter": iter,
        })
        self.best_metric_tracker.maybe_store(self.reg, val_metrics)
        wandb.log({
            "drr_image": wandb.Image(img.detach().squeeze()[None], mode="L", masks=masks),
            "gt_image": wandb.Image(self.gt_img.squeeze()[None], mode="L", masks=masks),
        })


# ── 训练循环 ────────────────────────────────────────────────────

def train(
    reg: Registration,
    gt_img_resampled: Tensor,
    gt_img: Tensor,
    optim: torch.optim.Optimizer,
    loss_fn: Callable,
    n_itrs: int,
    val_intervals: int,
    downsample_stride: int = 1,
    stage_name: str = "fine",
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    save_dir: Optional[str] = None,
) -> tuple[dict[ValidMetricKeys, float], Tensor, Tensor]:
    """单阶段训练循环。

    Args:
        reg: Registration 实例。
        gt_img_resampled: 下采样后的真实图像（用于损失计算）。
        gt_img: 原始分辨率真实图像（用于验证）。
        optim: 优化器。
        loss_fn: 损失函数 (pred, target) -> Tensor。
        n_itrs: 本阶段迭代次数。
        val_intervals: 验证间隔。
        downsample_stride: 投影下采样步长。
        stage_name: 阶段名（用于日志）。
        scheduler: 学习率调度器（可选）。
        save_dir: 结果保存目录（可选）。

    Returns:
        (final_metrics, best_rotations, best_translations)
    """
    from pathlib import Path
    from PIL import Image
    from visualization import save_label_overlay, save_masks_only

    wandb.watch(reg.drr)
    amp_enabled = False

    def _make_record(iter_: int, reg_: Registration, loss_: Tensor) -> dict:
        alpha, beta, gamma = reg_.rotation.squeeze().tolist()
        bx, by, bz = reg_.translation.squeeze().tolist()
        return {
            "loss": float(loss_.item()),
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "bx": bx, "by": by, "bz": bz,
            "stage": stage_name,
        }

    valid_fn = Validation(stage_name, reg, gt_img, loss_fn, BestMetricTracker())

    # 训练前验证 + 保存初始结果
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        img_val, ao_mask, lv_mask = run_with_masks(reg)
        save_label_overlay(gt_img, ao_mask, lv_mask, save_path / "01_init_gt_mask.png", alpha=0.7)
        save_masks_only(ao_mask, lv_mask, save_path / "01_init_mask.png")
        Image.fromarray(
            (img_val.squeeze().cpu().detach().numpy() * 255).astype("uint8")
        ).save(save_path / "01_init_drr.png")

    valid_fn(0)

    save_point_indices = {n_itrs * i // 10: i + 2 for i in range(1, 10)}

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

        if scheduler is not None:
            scheduler.step()

        current_lr_rot = optim.param_groups[0]["lr"]
        current_lr_trans = optim.param_groups[1]["lr"]

        pbar.set_description(f"[{stage_name}] loss = {loss.item():06f}")
        record = _make_record(itr, reg, loss.detach())
        record["lr_rotations"] = current_lr_rot
        record["lr_translations"] = current_lr_trans
        wandb.log(record)

        # 定时验证
        if itr >= val_intervals and itr % val_intervals == 0:
            valid_fn(itr)

        # 定时保存
        if save_dir is not None and itr in save_point_indices:
            save_idx = save_point_indices[itr]
            img_val, ao_mask, lv_mask = run_with_masks(reg)

            masks = {
                "labels": {
                    "mask_data": (ao_mask.to(torch.uint8) + lv_mask.to(torch.uint8) * 2),
                    "class_labels": {1: "AO", 2: "LV"},
                }
            }
            wandb.log({
                "drr_image": wandb.Image(img_val.detach().squeeze()[None], mode="L", masks=masks),
                "gt_image": wandb.Image(gt_img.squeeze()[None], mode="L", masks=masks),
            })

            save_path = Path(save_dir)
            prefix = f"{save_idx:02d}_iter{itr:04d}"
            save_label_overlay(gt_img, ao_mask, lv_mask, save_path / f"{prefix}_gt_mask.png", alpha=0.7)
            save_masks_only(ao_mask, lv_mask, save_path / f"{prefix}_mask.png")
            Image.fromarray(
                (img_val.squeeze().cpu().detach().numpy() * 255).astype("uint8")
            ).save(save_path / f"{prefix}_drr.png")

    # 保存最终结果
    if save_dir is not None:
        save_path = Path(save_dir)
        img_final, ao_mask_final, lv_mask_final = run_with_masks(reg)
        save_label_overlay(gt_img, ao_mask_final, lv_mask_final, save_path / "10_final_gt_mask.png", alpha=0.7)
        save_masks_only(ao_mask_final, lv_mask_final, save_path / "10_final_mask.png")
        Image.fromarray(
            (img_final.squeeze().cpu().detach().numpy() * 255).astype("uint8")
        ).save(save_path / "10_final_drr.png")

    best_score = valid_fn.best_metric_tracker.best_score
    best_rotations, best_translations = valid_fn.best_pose
    final_metrics = valid_fn.best_metrics

    print(
        f"[stage-summary] {stage_name}: "
        f"best_score={best_score:.6f}, "
        f"best_mse={final_metrics.get(ValidMetricKeys.MSE, float('nan')):.6f}, "
        f"best_ncc={final_metrics.get(ValidMetricKeys.NCC, float('nan')):.6f}"
    )
    wandb.log({
        "stage": stage_name,
        "stage/best_score": best_score,
        "stage/best_mse": final_metrics.get(ValidMetricKeys.MSE, float("nan")),
        "stage/best_ncc": final_metrics.get(ValidMetricKeys.NCC, float("nan")),
    })

    return final_metrics, best_rotations, best_translations
