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
