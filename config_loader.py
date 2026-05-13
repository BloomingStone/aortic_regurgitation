"""配置加载工具。"""

from __future__ import annotations

from pathlib import Path

import yaml
from easydict import EasyDict


def load_config(path: Path) -> EasyDict:
    """从 YAML 文件加载配置为 EasyDict。

    Args:
        path: YAML 配置文件路径。

    Returns:
        EasyDict 格式的配置对象（支持点号访问）。
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)
