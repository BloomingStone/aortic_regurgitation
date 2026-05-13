"""升主动脉投影配准 — 主入口。

将 CTA 数据使用 DRR 投影到 2D，与真实 X 片配准，优化投影位姿。
使用相同投影位姿，将主动脉和左心室的标签也投影到 2D X 片上，供下游生理指标计算。

用法：
    pixi run python optimize_pose.py
"""

from __future__ import annotations

from math import radians
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import wandb

from config_loader import load_config
from common_types import CArmGeometry, DrrSetting, GridSearchConfig, StageConfig
from geometry import pose_from_carm
from drr import get_drr
from losses import get_loss_fn, validation_score
from training import train


def main() -> None:
    config_path = Path("config/base_config.yaml")
    cfg: Any = load_config(config_path)

    wandb.init(
        project="aortic_regurgitation",
        config=cfg,
    )

    # ── 构造 DRR ────────────────────────────────────────────────
    geom = CArmGeometry(
        sdd=cfg.geom.sdd,
        sod=cfg.geom.sod,
        height=cfg.geom.height,
        delx=cfg.geom.delx,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    drr = get_drr(
        Path(cfg.data.img_path),
        Path(cfg.data.seg_path),
        Path(cfg.data.whole_heart_label_path),
        geom,
        device,
        config=DrrSetting(**cfg.get("drr", {})),
    ).eval()

    # ── 加载 2D 真实图像 ────────────────────────────────────────
    img_2d = T.ToTensor()(
        T.Resize((cfg.geom.height, cfg.geom.height))(
            Image.open(cfg.data.image_2d_path).convert("L")
        )
    )[None].to(device)

    # ── 构造损失函数 ────────────────────────────────────────────
    loss_fn = get_loss_fn(cfg.loss, device)

    # (可选) 粗搜索初始化位姿
    rotations, translations = pose_from_carm(
        geom.sod, 0, 0,
        radians(cfg.init_pose.alpha), radians(cfg.init_pose.beta), 0.,
    )

    # ── 构造 Registration ───────────────────────────────────────
    from diffdrr.registration import Registration
    reg = Registration(
        drr,
        rotations.to(device),
        translations.to(device),
        parameterization="euler_angles",
        convention="ZXY",
    )

    # ── 构造优化器 ──────────────────────────────────────────────
    optim_type = cfg.optimizer.type.upper()
    if optim_type == "SGD":
        optim_cls = torch.optim.SGD
    elif optim_type == "ADAM":
        optim_cls = torch.optim.Adam
    else:
        raise NotImplementedError(f"Unknown optimizer type: {optim_type}")

    optim = optim_cls(
        [
            {"params": [reg._rotation], "lr": cfg.optimizer.lr_rotations},
            {"params": [reg._translation], "lr": cfg.optimizer.lr_translations},
        ],
        **cfg.optimizer.init_args,
    )

    # ── 构造训练阶段 ────────────────────────────────────────────
    coarse_cfg = cfg.get("coarse_to_fine", None)
    if coarse_cfg is not None and coarse_cfg.get("stages", None):
        stages = [StageConfig.from_dict(s) for s in coarse_cfg.stages]
    else:
        total_itrs = int(cfg.train.n_itrs)
        coarse_itrs = max(1, total_itrs // 3)
        mid_itrs = max(1, total_itrs // 3)
        fine_itrs = max(1, total_itrs - coarse_itrs - mid_itrs)
        stages = [
            StageConfig("coarse", downsample_stride=4, n_itrs=coarse_itrs, lr_rot_mult=0.25, lr_trans_mult=0.25),
            StageConfig("mid", downsample_stride=2, n_itrs=mid_itrs, lr_rot_mult=0.5, lr_trans_mult=0.5),
            StageConfig("fine", downsample_stride=1, n_itrs=fine_itrs, lr_rot_mult=1.0, lr_trans_mult=1.0),
        ]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim,
        milestones=[stages[0].n_itrs, stages[0].n_itrs + stages[1].n_itrs],
        gamma=0.5,
    )

    # ── 逐阶段训练 ──────────────────────────────────────────────
    stage_summaries: list[dict[str, Any]] = []
    for stage in stages:
        if stage.downsample_stride > 1:
            gt_img_resampled = F.avg_pool2d(
                img_2d,
                kernel_size=stage.downsample_stride,
                stride=stage.downsample_stride,
            )
        else:
            gt_img_resampled = img_2d

        best_metrics, _, _ = train(
            reg=reg,
            gt_img_resampled=gt_img_resampled,
            gt_img=img_2d,
            optim=optim,
            loss_fn=loss_fn,
            n_itrs=stage.n_itrs,
            val_intervals=cfg.train.val_interval,
            downsample_stride=stage.downsample_stride,
            stage_name=stage.name,
            scheduler=scheduler,  # type: ignore
            save_dir=str(Path("results") / stage.name),
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
