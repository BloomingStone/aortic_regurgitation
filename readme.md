# 升主动脉配准

使用[diffdrr](https://github.com/eigenvivek/DiffDRR)进行升主动脉配准，以进行后续计算闭合率等指标。

## 环境配置

此项目使用 [pixi](https://pixi.sh/) 配置环境，可能需要从下载pixi: https://pixi.sh/

使用 `pixi install --frozen` 从 `pixi.lock` 安装环境。该方法确保环境的一致性，避免因依赖版本变化引起的问题。

如果安装失败，可能是由于某些依赖包在当前环境中无法安装。可以尝试以下步骤：
+ 使用 `pixi install` 重新安装，这将允许pixi根据当前环境自动调整依赖版本。
+ 如果仍然无法安装，检查错误日志，确定是哪个包导致安装失败，并修改 `pixi.toml` 中对应包的版本

安装完成后，可以使用 `pixi shell` 进入环境，或者使用 `pixi run <command>` 直接运行命令，例如 `pixi run python generate_4d_dvf.py`。

log 默认使用 wandb 可以参考[官网](https://wandb.ai/)进行注册登录和创建项目API

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
