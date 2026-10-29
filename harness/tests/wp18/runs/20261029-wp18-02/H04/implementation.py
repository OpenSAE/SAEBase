"""规范化显式 synthetic 的设备记录。嵌套密钥和路径不能进入结果。"""

from __future__ import annotations


class EnvironmentRefusal(ValueError):
    """记录不能被完整检查，因此不会返回部分结果。"""


_SECRET_KEYS = frozenset({"token", "secret", "password", "api_key", "apikey", "authorization"})
_MAX_DEPTH = 4


def normalize_environment(record: dict) -> dict:
    """把 synthetic 记录收成未观测概述。

    字典和列表最多展开四层。无法检查的类型会整份拒绝。
    空字段写成 unknown。没有探针输出不会被写成设备不存在。
    """
    if not isinstance(record, dict):
        raise EnvironmentRefusal("record must be an object")
    if record.get("synthetic") is not True:
        raise EnvironmentRefusal("synthetic true is required")
    body = {key: value for key, value in record.items() if key != "synthetic"}
    redacted: list[str] = []
    fields = _walk(body, 1, "$", redacted)
    if not isinstance(fields, dict):
        raise EnvironmentRefusal("record body must stay an object")
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
        "summary_zh": "这是 synthetic 测验记录，没有探测真实设备，不能当作机器实测或研究任务成功。",
        "summary_en": "Synthetic exam record. No real device was probed, so this is not a measurement or a completed research task.",
    }


def _walk(value: object, depth: int, trail: str, redacted: list[str]) -> object:
    """检查一层容器。超过深度或不支持的类型时失败，不把原值传出去。"""
    if depth > _MAX_DEPTH:
        raise EnvironmentRefusal(f"unsupported depth at {trail}")
    if isinstance(value, dict):
        cleaned = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise EnvironmentRefusal(f"unsupported key at {trail}")
            child = f"{trail}.{key}"
            if _sensitive_key(key) or _private_text(value[key]):
                redacted.append(child)
                cleaned[key] = "[redacted]"
                continue
            cleaned[key] = _walk(value[key], depth + 1, child, redacted)
        return cleaned
    if isinstance(value, list):
        return [_walk(item, depth + 1, f"{trail}[{index}]", redacted) for index, item in enumerate(value)]
    if isinstance(value, str):
        return _public_text(value)
    if value is None:
        return "unknown"
    raise EnvironmentRefusal(f"unsupported value at {trail}")


def _sensitive_key(key: str) -> bool:
    """按字段名识别密钥和私有路径槽位。"""
    return key.lower() in _SECRET_KEYS or key.lower() in {"path", "home", "cwd", "user_path", "hostname"}


def _private_text(value: object) -> bool:
    """绝对路径字符串即使放在普通字段里也不进入结果。"""
    return isinstance(value, str) and (value.startswith("/") or "\\" in value)


def _public_text(value: str) -> str:
    """空字符串和探针字样都保持 unknown，不用它们推断没有 GPU。"""
    if value == "":
        return "unknown"
    lowered = value.lower()
    if "nvidia-smi" in lowered or lowered in {"none", "false", "no gpu", "cpu only"}:
        return "unknown"
    return value
