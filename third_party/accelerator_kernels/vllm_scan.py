# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Copied from vLLM 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac,
# vllm/third_party/flash_linear_attention/ops/cumsum.py (original FLA MIT).
# Kernel body unchanged; use one fixed warp instead of batch-dependent autotuning.
"""Fixed-row upstream cumulative scan for Qwen GDN's 64-element decay chunks."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def stable_chunk_cumsum(value):
    if value.device.type != "cuda":
        return value.cumsum(dim=-1)
    if value.dtype != torch.float32:
        raise ValueError("Qwen decay scan expects FP32")
    values = value.contiguous()
    width = values.shape[-1]
    rows = values.numel() // width
    output = torch.empty_like(values)
    chunk_local_cumsum_scalar_kernel[(1, rows)](
        values,
        output,
        None,
        None,
        width,
        B=rows,
        H=1,
        BT=triton.next_power_of_2(width),
        REVERSE=False,
        IS_VARLEN=False,
        HEAD_FIRST=True,
        num_warps=1,
    )
    return output
