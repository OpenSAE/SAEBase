"""One numerical policy per GPU worker, established before model loading.

The CUDA/cuBLAS policy is adapted from vLLM v0.27.1, Apache-2.0,
commit 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac,
vllm/model_executor/layers/batch_invariant.py. GPU kernels are copied from upstream; no new numerical algorithm is added.
Tasks own their numerical checks; this module owns process-wide execution.
"""

import os
import threading

import torch

_LOCK = threading.RLock()
_PROFILE = None
_LIBRARY = None
_MODEL_PROTOCOL = None
SOURCE = "https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/layers/batch_invariant.py"
POLICY = "saebase.vllm_batch_invariant.v4"
QWEN_POLICY = "saebase.vllm_batch_invariant.qwen_gdn_scan.v5"
SOFTMAX_SOURCE = "https://github.com/vllm-project/vllm/blob/68e478651a20e3d23bf939912c38883d511ac3ee/vllm/model_executor/determinism/batch_invariant.py"


def _softmax(input, dim, half_to_float=False):
    """Preserve native CUDA's FP32 accumulation for FP16/BF16 probabilities."""
    from saebase._vendor.vllm_ops import softmax_batch_invariant

    output_dtype = torch.float32 if half_to_float else input.dtype
    values = input.float() if input.dtype in {torch.float16, torch.bfloat16} else input
    return softmax_batch_invariant(values, dim).to(output_dtype)


def _log_softmax(input, dim, half_to_float=False):
    """Match native accumulation/output dtypes around the copied vLLM operation."""
    from saebase._vendor.vllm_ops import log_softmax

    output_dtype = torch.float32 if half_to_float else input.dtype
    values = input.float() if input.dtype in {torch.float16, torch.bfloat16} else input
    return log_softmax(values, dim).to(output_dtype)


def configure(profile=None, *, device="cuda"):
    """Configure once, before the first CUDA operation; never mix policies in a worker."""
    global _PROFILE, _LIBRARY
    if not str(device).startswith("cuda"):
        return identity()
    profile = profile or _PROFILE or "native"
    if profile not in {"native", "vllm"}:
        raise ValueError("runtime.numerics must be native or vllm")
    with _LOCK:
        if _PROFILE is not None:
            if _PROFILE != profile:
                raise RuntimeError(
                    "Numerical policy cannot change in a loaded worker; start a new worker"
                )
            return identity()
        if profile == "vllm":
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "Select runtime.numerics=vllm before CUDA/model initialization; start a new worker"
                )
            # vLLM disables split-K on this cuBLAS path, fixes the BLAS library,
            # disables reduced-precision reductions and requests IEEE FP32.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (False, False)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
            torch.backends.cuda.preferred_blas_library(backend="cublaslt")
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.rnn.fp32_precision = "ieee"
            # HF uses PyTorch SDPA, not vLLM's attention kernels. Select its
            # public math backend to remove batch-dependent fused dispatch.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            # vLLM also fixes vocabulary normalization. Native log-softmax
            # can differ by one FP32 ULP even when the logits are identical.
            from saebase._vendor import vllm_ops

            _LIBRARY = torch.library.Library("aten", "IMPL")
            for operation, implementation in {
                "mm": vllm_ops.mm_batch_invariant,
                "bmm": vllm_ops.bmm_batch_invariant,
                "addmm": vllm_ops.addmm_batch_invariant,
                "linear": vllm_ops.linear_batch_invariant,
                "_log_softmax": _log_softmax,
                "_softmax": _softmax,
            }.items():
                _LIBRARY.impl("aten::" + operation, implementation, "CUDA")
        _PROFILE = profile
        return identity()


def identity():
    """Record actual worker policy alongside results, not as a model weight identity."""
    return {
        "profile": _PROFILE or "native",
        "protocol_id": (_MODEL_PROTOCOL or POLICY)
        if _PROFILE == "vllm"
        else "saebase.native_numerics.v1",
        "upstream": SOURCE if _PROFILE == "vllm" else None,
        "upstream_softmax": SOFTMAX_SOURCE if _PROFILE == "vllm" else None,
        **({"qwen_gdn_scan": "vllm_fixed_row_cumsum.v1"} if _MODEL_PROTOCOL == QWEN_POLICY else {}),
        "operators": ["vllm_matmul", "vllm_bmm", "vllm_softmax", "torch_sdpa_math"]
        if _PROFILE == "vllm"
        else [],
        "scope": "single GPU Torch/Triton execution; task precheck validates actual batch invariance",
    }


def configure_model(model, *, device):
    """Install the copied stable GDN scan only for Qwen GPU workers."""
    if _PROFILE != "vllm" or not str(device).startswith("cuda"):
        return
    if not str(getattr(getattr(model, "config", None), "model_type", "")).startswith("qwen3_5"):
        return
    import transformers.models.qwen3_5.modeling_qwen3_5 as upstream
    from saebase._vendor.hf_qwen_gdn import torch_chunk_gated_delta_rule

    upstream.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    global _MODEL_PROTOCOL
    _MODEL_PROTOCOL = QWEN_POLICY


def configure_task(cfg):
    """Keep configuration-only invocations free of execution side effects."""
    if not cfg.get("task", {}).get("execute", False):
        return identity()
    # A saved pre-policy configuration is an explicit legacy run. New Hydra
    # configurations declare vllm; do not reinterpret old resume identities.
    profile = cfg.get("runtime", {}).get("numerics", "native")
    from models.loading_policy import resolve_device

    device = resolve_device(cfg.task.get("device", cfg.get("model", {}).get("device", "auto")))
    return configure(str(profile), device=device)


def bind_configuration(cfg):
    """Persist the semantic revision before Runtime checks resume compatibility."""
    runtime = cfg.get("runtime", {})
    if runtime.get("numerics") == "vllm":
        from omegaconf import open_dict

        with open_dict(runtime):
            name = str(cfg.get("model", {}).get("name", ""))
            runtime.numerics_protocol = QWEN_POLICY if name.startswith("qwen35_") else POLICY
