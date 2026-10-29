"""规范化显式 synthetic 的设备记录。不探测 GPU，也不保存真实机器信息。"""

from __future__ import annotations


class EnvironmentRecordError(ValueError):
    """记录不是允许进入测验的合成设备说明。"""


_SECRET_KEYS = frozenset({"token", "secret", "password", "api_key", "apikey", "authorization"})
_PATH_KEYS = frozenset({"path", "home", "cwd", "user_path", "hostname"})


def normalize_environment(record: dict) -> dict:
    """把 synthetic 设备 JSON 收成可审阅概述。

    缺字段写成 unknown。token、secret 和私有路径只保留字段名。
    没有 nvidia-smi 输出不会被写成“无 GPU”。
    """
    if not isinstance(record, dict) or record.get("synthetic") is not True:
        raise EnvironmentRecordError("only an object with synthetic true is accepted")
    redacted: list[str] = []
    fields: dict[str, object] = {}
    for key in sorted(record):
        if key == "synthetic":
            continue
        value = record[key]
        if _secret(key, value):
            redacted.append(key)
            continue
        fields[key] = _public_value(key, value)
    if "gpu" not in fields:
        fields["gpu"] = "unknown"
    return {
        "synthetic": True,
        "observed": False,
        "measurement": "not_observed",
        "source": "synthetic-fixture",
        "gpu": fields["gpu"],
        "fields": fields,
        "redacted_fields": redacted,
        "summary_zh": "这是 synthetic 测验记录，没有探测真实设备，不能当作研究任务成功或机器实测。",
        "summary_en": "Synthetic exam record. No real device was probed, so this is not a measurement or a completed research task.",
    }


def _secret(key: str, value: object) -> bool:
    """识别应从产物中去掉的密钥和私有路径。"""
    lowered = key.lower()
    if lowered in _SECRET_KEYS or lowered in _PATH_KEYS:
        return True
    if isinstance(value, str) and (value.startswith("/") or "\\" in value):
        return True
    return False


def _public_value(key: str, value: object) -> object:
    """空值和探针失败都保持 unknown，不推断硬件不存在。"""
    if value is None or value == "":
        return "unknown"
    if isinstance(value, str) and "nvidia-smi" in value.lower():
        return "unknown"
    if key == "gpu" and isinstance(value, str) and value.lower() in {"none", "false", "no gpu", "cpu only"}:
        return "unknown"
    return value
