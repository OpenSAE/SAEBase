"""目的和主要功能：独立读取本项目的常规或密集开发选择。

适用范围：项目 Hook 与会话提示；密集开发的读取使用标准库，科研开发可独立于中央安装。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def current_mode(project_root=ROOT):
    path = Path(project_root) / ".researchharness/mode.json"
    if not path.exists():
        return "regular"
    if path.is_symlink():
        raise ValueError("Project mode must be a regular local file")
    value = json.loads(path.read_text())
    if value.get("schema") != "rh.development_mode" or value.get("mode") not in {
        "regular",
        "intensive",
    }:
        raise ValueError("Invalid project development mode")
    return value["mode"]
