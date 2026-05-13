# 升主动脉配准

使用[diffdrr](https://github.com/eigenvivek/DiffDRR)进行升主动脉配准，以进行后续计算闭合率等指标。

## 环境配置

此项目使用 [pixi](https://pixi.sh/) 配置环境，可能需要从下载pixi: https://pixi.sh/

使用 `pixi install --frozen` 从 `pixi.lock` 安装环境。该方法确保环境的一致性，避免因依赖版本变化引起的问题。

如果安装失败，可能是由于某些依赖包在当前环境中无法安装。可以尝试以下步骤：
+ 使用 `pixi install` 重新安装，这将允许pixi根据当前环境自动调整依赖版本。
+ 如果仍然无法安装，检查错误日志，确定是哪个包导致安装失败，并修改 `pixi.toml` 中对应包的版本

安装完成后，可以使用 `pixi shell` 进入环境，或者使用 `pixi run <command>` 直接运行命令。

log 默认使用 wandb 可以参考[官网](https://wandb.ai/)进行注册登录和创建项目API

## 项目结构

```
├── optimize_pose.py   # 主入口，组装流程
├── common_types.py    # 类型别名、枚举、dataclass（LabelID, CArmGeometry, StageConfig 等）
├── geometry.py        # 几何/位姿/仿射变换（pose_from_carm, recenter 等）
├── preprocessing.py   # 体数据加载、裁剪 ROI、对比度调整
├── drr.py             # DRR 构造与投影（get_drr, project_image, run_with_masks 等）
├── losses.py          # 损失函数与验证指标（get_loss_fn 支持多损失加权）
├── training.py        # 训练循环、验证、学习率调度器
├── coarse_init.py     # 粗搜索初始化位姿（alpha/beta/sod 三维网格搜索）
├── visualization.py   # 图像保存与标签叠加（save_label_overlay, save_masks_only）
├── config_loader.py   # YAML 配置加载
├── transforms.py      # MedicalImage, ClipROITransform, ResampleTransform
├── config/
│   └── base_config.yaml
└── pixi.toml
```

各模块功能明确，高内聚低耦合，便于独立理解、测试和迭代改进。

## 训练

编辑 `config/base_config.yaml` 控制实验配置, 如训练步长、学习率衰减、所用loss等。

使用 `pixi run python optimize_pose.py` 启动投影配准训练。

## 初始化参数

`base_config.yaml` 中的投影几何和初始化位姿可以从 DSA dcm 文件中提取，对应关系如下：
- `init_pose.alpha`: 主角度（单位：度），`0018,1510 Positioner Primary Angle`
- `init_pose.beta`: 次角度（单位：度），`0018,1511 Positioner Secondary Angle`
- `geom.sdd`: 射线源到探测器的距离（单位：mm），`0018,1110 Distance Source to Detector`
- `geom.sod`: 射线源到物体的距离（单位：mm），`0018,1111 Distance Source to Patient`
- `geom.height`: 探测器所得图像高度（单位：pixel），`0028,0010 Rows`
- `geom.delx`: 探测器像素间距（单位：mm/pixel），`0018,1164 Imager Pixel Spacing`

注意，目前仅支持图像高度和宽度相同，且像素间距相同的情况，未使用到 `0028,0011 Columns`

### 初始化参数对结果的影响

可能需要调整初始化参数，使其投影结果与真实图像尽可能接近，并适当提高视角从而能看到更多组织。后续位姿优化效果才相对较好

## 核心流程

1. **加载 CTA 体数据** — 从 NIfTI 加载 3D 图像及主动脉/左心室/全心分割标签
2. **预处理** — 裁剪 ROI → 重采样降分辨率 → 重新居中 → 碘造影剂对比度调整
3. **构造 DRR** — 基于 C-arm 几何参数（`CArmGeometry`）和 `diffdrr` 渲染器
4. **初始化位姿** — 从 DSA 参数直接计算，可选 `coarse_init_pose` 粗搜索
5. **配准优化** — 粗→中→细三阶段逐级优化，每阶段降低下采样倍率
6. **投影标签** — 使用优化得到的位姿将主动脉/左心室标签投影到 2D X 片

https://wandb.ai/bloomingstone-southeast-university/aortic_regurgitation/reports/---VmlldzoxNjg2NTk1OQ?accessToken=vcgh7ukovopljng5hz2mwshciv67iy0mywzvhzaexal5wxgl9st5e7bz26weawem