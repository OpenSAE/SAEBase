"""vLLM 启动参数。端口、显存比例、FlashInfer 开关只改这里。"""

from __future__ import annotations

import os
from pathlib import Path

HOST = "127.0.0.1"
PORT = 18000
DTYPE = "bfloat16"
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def apply_process_env() -> None:
    """当前解释器拉起 vLLM 之前要设的环境变量。"""
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
