#!/usr/bin/env python3
"""NNsight-vLLM worker。用当前解释器以 ``python -m tasks.generate.backend.nnsight_worker`` 启动。

vLLM 的 spawn 要求本文件作为 ``__main__``。不要在别的进程里 import 之后直接建引擎。
模型名和层号由调用方传入，这里不写默认模型。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_model(args: argparse.Namespace, blob: dict | None = None) -> str:
    model_id = ""
    if blob is not None:
        model_id = str(blob.get("model_id") or "")
    model_id = model_id or str(getattr(args, "model_id", "") or "")
    if not model_id:
        raise ValueError("model_id is required")
    return model_id


def _require_layer(args: argparse.Namespace, blob: dict | None = None) -> int:
    if blob is not None and blob.get("layer") is not None:
        return int(blob["layer"])
    if getattr(args, "layer", None) is not None:
        return int(args.layer)
    raise ValueError("layer is required")


def _emit(payload: dict) -> None:
    print(json.dumps(payload, default=str), flush=True)


def _unpack_layer_out(out):
    """Last-token resid for decode-step mutation. Returns (x_last, hidden_ref, gemma_tuple)."""
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        hidden, residual = out[0], out[1]
        if (
            torch_is_tensor(hidden)
            and torch_is_tensor(residual)
            and tuple(hidden.shape) == tuple(residual.shape)
        ):
            # Full (B,T,D) or (T,D) tensors — take last position.
            summed = hidden + residual
            if summed.ndim >= 2:
                x = summed[..., -1, :] if summed.ndim == 3 else summed[-1]
            else:
                x = summed
            return x, hidden, True
        x = hidden[-1] + residual[-1]
        return x, hidden, True
    hidden = out[0] if isinstance(out, (tuple, list)) else out
    x = hidden[-1] if getattr(hidden, "ndim", 0) >= 2 else hidden
    return x, hidden, False


def torch_is_tensor(x) -> bool:
    import torch

    return isinstance(x, torch.Tensor)


def _unpack_full_sequence(out):
    """Full-sequence resid for Baseline encode dump. Returns (T, d) CPU-ready tensor."""
    import torch

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        a, b = out[0], out[1]
        if (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and tuple(a.shape) == tuple(b.shape)
        ):
            tens = (a + b).detach().float()
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            # Per-token lists (Gemma-style)
            toks = []
            for ha, hb in zip(a, b):
                toks.append((ha + hb).detach().float().reshape(-1))
            tens = torch.stack(toks, 0)
        else:
            tens = (a[-1] + b[-1]).detach().float()
            if tens.ndim == 1:
                return tens.unsqueeze(0)
            return tens
    else:
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(hidden, (list, tuple)):
            tens = torch.stack([h.detach().float().reshape(-1) for h in hidden], 0)
        else:
            tens = hidden.detach().float()
    if tens.ndim == 3:
        tens = tens[0]
    if tens.ndim == 1:
        tens = tens.unsqueeze(0)
    return tens


def _writeback(hidden, x, x2, gemma_tuple: bool) -> None:
    delta = (x2 - x).to(dtype=hidden.dtype if hasattr(hidden, "dtype") else x2.dtype)
    if gemma_tuple and torch_is_tensor(hidden) and hidden.ndim >= 2:
        # Tensor residual stream: write last position.
        if hidden.ndim == 3:
            hidden[..., -1, :] = hidden[..., -1, :] + delta
        elif hidden.ndim == 2:
            hidden[-1] = hidden[-1] + delta
        else:
            hidden += delta
    elif gemma_tuple:
        hidden[-1] += delta
    else:
        hidden[-1] += delta


def _decoder_tap(model_id: str, layer: int) -> str:
    """Module path for a decoder-layer residual tap.

    Gemma / Llama / Qwen3 use ``model.layers``. Qwen3.5-4B is
    ``Qwen3_5ForConditionalGeneration``: text residual lives under
    ``language_model.model.layers``. Same tensor, different wrapper.
    """
    arch = ""
    cfg = Path(str(model_id)) / "config.json"
    if cfg.is_file():
        try:
            arch = str(json.loads(cfg.read_text(encoding="utf-8")).get("architectures", [""])[0])
        except Exception:
            arch = ""
    if arch.startswith("Qwen3_5"):
        return f"language_model.model.layers.{int(layer)}.output"
    return f"model.layers.{int(layer)}.output"


def _is_qwen35(model_id: str) -> bool:
    cfg = Path(str(model_id)) / "config.json"
    if not cfg.is_file():
        return "Qwen3.5" in str(model_id) or "Qwen3_5" in str(model_id)
    try:
        arch = str(json.loads(cfg.read_text(encoding="utf-8")).get("architectures", [""])[0])
    except Exception:
        return False
    return arch.startswith("Qwen3_5")


def _layer_out(root, layer: int, *, qwen35: bool):
    """Residual output of decoder layer ``layer`` on the trace root envoy."""
    if qwen35:
        return root.language_model.model.layers[int(layer)].output
    return root.model.layers[int(layer)].output


def _build_engine(
    *,
    model_id: str,
    gpu_mem: float,
    max_model_len: int,
    taps: bool,
    layer: int,
    max_num_seqs: int = 16,
    extra: dict | None = None,
):
    from nnsight.modeling.vllm import VLLM

    kwargs: dict = dict(
        tensor_parallel_size=1,
        gpu_memory_utilization=float(gpu_mem),
        max_model_len=int(max_model_len),
        max_num_seqs=int(max_num_seqs),
        dispatch=True,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    if taps:
        kwargs["taps"] = [_decoder_tap(model_id, int(layer))]
    else:
        kwargs["enforce_eager"] = True
    if extra:
        kwargs.update(extra)
    return VLLM(model_id, **kwargs)


def cmd_smoke(args: argparse.Namespace) -> dict:
    model_id = _require_model(args)
    t_load = time.perf_counter()
    model = _build_engine(
        model_id=model_id,
        gpu_mem=args.gpu_mem,
        max_model_len=args.max_model_len,
        taps=False,
        layer=_require_layer(args),
    )
    load_s = time.perf_counter() - t_load
    prompt = "The capital of France is"
    t_gen = time.perf_counter()
    with model.trace(
        prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(args.max_tokens),
        ignore_eos=True,
    ) as tracer:
        result = tracer.result.save()
    gen_s = time.perf_counter() - t_gen
    text = result.outputs[0].text
    n_tok = len(result.outputs[0].token_ids)
    return {
        "ok": True,
        "phase": "smoke",
        "load_s": load_s,
        "gen_s": gen_s,
        "n_gen_tokens": n_tok,
        "tok_per_s": n_tok / max(gen_s, 1e-6),
        "text_preview": text[:200],
        "model": model_id,
        "eager": True,
    }


def _rows_from_state(state: dict):
    import torch

    _tm0 = __import__("importlib").import_module("tasks.10_cross_model_steering.gemmascope_rows")
    PairSteerRows = _tm0.PairSteerRows

    def t(key: str):
        v = state.get(key)
        return v if v is None else torch.as_tensor(v).float()

    return PairSteerRows(
        feature_i=int(state["feature_i"]),
        feature_j=int(state["feature_j"]),
        w_enc_i=t("w_enc_i"),
        w_enc_j=t("w_enc_j"),
        b_i=t("b_i"),
        b_j=t("b_j"),
        threshold_i=t("threshold_i"),
        threshold_j=t("threshold_j"),
        d_i=t("d_i"),
        d_j=t("d_j"),
        tau_i=float(state["tau_i"]),
        tau_j=float(state["tau_j"]),
        subtract_b_dec=bool(state["subtract_b_dec"]),
        b_dec=t("b_dec"),
        gate=str(state.get("gate") or "pre"),
        operator=str(state.get("operator") or "threshold"),
        alpha=float(state.get("alpha") if state.get("alpha") is not None else 1.0),
        v=t("v"),
    )


def _load_sae_pack(path):
    if not path:
        return None
    import torch

    pack = torch.load(str(path), map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec"):
        val = pack.get(key)
        if isinstance(val, torch.Tensor):
            pack[key] = val.to(device=device, dtype=torch.float32)
    if isinstance(pack.get("w_enc"), torch.Tensor):
        pack["w_enc_t"] = pack["w_enc"].T.contiguous()
        pack.pop("w_enc", None)
    return pack


def _keep_mask(spec, job: dict, d_sae: int, device):
    import torch

    mask = torch.zeros(int(d_sae), device=device, dtype=torch.bool)
    ids = list((spec.feature_ids if spec is not None else None) or job.get("feature_ids") or [])
    if ids:
        mask[torch.as_tensor(ids, device=device, dtype=torch.long)] = True
    return mask


def _edit_hidden(hidden, spec, mode: str, sae_pack, job: dict, keep_mask=None):
    _tm1 = __import__("importlib").import_module("tasks.13_functional_concept_validation.sae_edit")
    apply_ablate = _tm1.apply_ablate

    if mode == "ablate":
        return apply_ablate(hidden, spec)
    if sae_pack is None:
        raise RuntimeError("keep requires sae_pack")
    import torch

    x = hidden.float()
    b_dec = sae_pack.get("b_dec")
    if b_dec is not None:
        x = x - b_dec
    w_t = sae_pack.get("w_enc_t")
    if w_t is None:
        w_t = sae_pack["w_enc"].T.contiguous()
        sae_pack["w_enc_t"] = w_t
    pre = x @ w_t + sae_pack["b_enc"]
    base = torch.relu(pre)
    thr = sae_pack["threshold"]
    if str(sae_pack.get("gate") or "pre") == "relu":
        z = base * (base > thr).to(base.dtype)
    else:
        z = base * (pre > thr).to(base.dtype)
    if keep_mask is None:
        keep_mask = _keep_mask(spec, job, z.shape[-1], z.device)
    if z.ndim == 2 and keep_mask.ndim == 1:
        keep_mask = keep_mask.unsqueeze(0)
    z_prime = torch.where(keep_mask, z, z.new_zeros(z.shape))
    delta = (z_prime - z) @ sae_pack["w_dec"]
    return (hidden.float() + delta).to(dtype=hidden.dtype)


def _score_vec(vec, job: dict):
    ids = [int(x) for x in (job.get("option_token_ids") or [])]
    gold = int(job.get("gold_index") or 0)
    if not ids:
        return None
    scores = [float(vec[i]) for i in ids]
    others = [v for i, v in enumerate(scores) if i != gold]
    return scores[gold] - (max(others) if others else 0.0)


def _margin_from_saved(cap, index: int, job: dict):
    info = {"margin": None, "margin_error": None, "margin_step_hit": False, "n_captured": 0}
    hit = Path("/workspace/SAEReasoning/experiment/tmp/13_margin_logit.pt")
    errp = Path("/workspace/SAEReasoning/experiment/tmp/13_margin_logit_err.txt")
    if hit.is_file():
        import torch

        vec = torch.load(hit, map_location="cpu", weights_only=False).float().reshape(-1)
        info["margin_step_hit"] = True
        info["n_captured"] = 1
        info["margin"] = _score_vec(vec, job)
        return info
    if errp.is_file():
        info["margin_error"] = errp.read_text(encoding="utf-8")
        return info
    if cap is None or not list(cap or []):
        info["margin_error"] = "logit capture empty"
        return info
    logits = cap[-1]
    import torch

    if not isinstance(logits, torch.Tensor):
        info["margin_error"] = f"logits type {type(logits)}"
        return info
    vec = logits.detach().float().reshape(-1, logits.shape[-1])[-1].cpu()
    info["n_captured"] = len(list(cap))
    info["margin_step_hit"] = True
    info["margin"] = _score_vec(vec, job)
    return info


def _edit_generated_span(out, spec, n_prompt: int) -> None:
    """Ablate selected features on generated positions of a prefill residual."""
    _tm2 = __import__("importlib").import_module("tasks.13_functional_concept_validation.sae_edit")
    apply_ablate = _tm2.apply_ablate

    if spec is None:
        return
    n_prompt = max(0, int(n_prompt))

    def _edit_tensor(hidden, residual):
        summed = hidden + residual
        if summed.ndim == 3:
            gen = summed[0, n_prompt:]
            if gen.shape[0] <= 0:
                return
            x2 = apply_ablate(gen, spec)
            delta = (x2 - gen).to(dtype=hidden.dtype)
            hidden[0, n_prompt:] = hidden[0, n_prompt:] + delta
            return
        if summed.ndim == 2:
            gen = summed[n_prompt:]
            if gen.shape[0] <= 0:
                return
            x2 = apply_ablate(gen, spec)
            delta = (x2 - gen).to(dtype=hidden.dtype)
            hidden[n_prompt:] = hidden[n_prompt:] + delta

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        hidden, residual = out[0], out[1]
        if (
            torch_is_tensor(hidden)
            and torch_is_tensor(residual)
            and tuple(hidden.shape) == tuple(residual.shape)
        ):
            _edit_tensor(hidden, residual)
            return
        if isinstance(hidden, (list, tuple)) and isinstance(residual, (list, tuple)):
            for t in range(n_prompt, min(len(hidden), len(residual))):
                x = hidden[t] + residual[t]
                x2 = apply_ablate(x, spec)
                hidden[t] = hidden[t] + (x2 - x).to(dtype=hidden[t].dtype)
            return
    hidden = out[0] if isinstance(out, (tuple, list)) else out
    if torch_is_tensor(hidden) and hidden.ndim >= 2:
        gen = hidden[..., n_prompt:, :] if hidden.ndim == 3 else hidden[n_prompt:]
        if gen.numel() == 0:
            return
        x2 = apply_ablate(gen.reshape(-1, gen.shape[-1]), spec).reshape(gen.shape)
        if hidden.ndim == 3:
            hidden[..., n_prompt:, :] = hidden[..., n_prompt:, :] + (x2 - gen).to(hidden.dtype)
        else:
            hidden[n_prompt:] = hidden[n_prompt:] + (x2 - gen).to(hidden.dtype)


def _pin_spec(spec, device):
    import torch

    if spec is None:
        return None
    spec.w_enc = spec.w_enc.to(device=device, dtype=torch.float32)
    spec.b_enc = spec.b_enc.to(device=device, dtype=torch.float32)
    spec.threshold = spec.threshold.to(device=device, dtype=torch.float32)
    spec.w_dec = spec.w_dec.to(device=device, dtype=torch.float32)
    if spec.b_dec is not None:
        spec.b_dec = spec.b_dec.to(device=device, dtype=torch.float32)
    return spec


def _teacher_force_ids(model, job: dict) -> tuple[list[int], int]:
    tok = getattr(model, "tokenizer", None)
    forced = [int(x) for x in (job.get("forced_ids") or [])]
    margin_step = max(0, int(job.get("margin_step") or 0))
    prefix = forced[:margin_step]
    if job.get("input_ids"):
        ids = [int(x) for x in job["input_ids"]]
        n_prompt = int(job.get("n_prompt") or max(0, len(ids) - len(prefix)))
        return ids, n_prompt
    prompt = str(job.get("prompt") or "")
    if tok is None:
        raise RuntimeError("teacher_force prefill needs a tokenizer")
    prompt_ids = list(tok.encode(prompt, add_special_tokens=False))
    return prompt_ids + prefix, len(prompt_ids)


def _teacher_force_trace(model, jobs, *, layer, qwen35, load_s, chunk_n: int = 16) -> dict:
    """Prefill prompt+continuation[:margin_step], ablate generated span, read next-token logits.

    One forward per job instead of token-by-token decode. Batched via tracer.invoke.
    """
    import nnsight
    import torch

    _tm3 = __import__("importlib").import_module("tasks.13_functional_concept_validation.sae_edit")
    SetEditSpec = _tm3.SetEditSpec

    out_rows = []
    t_gen = time.perf_counter()
    chunk_n = max(1, int(chunk_n))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prog = Path("/workspace/SAEReasoning/experiment/tmp/13_tf_progress.jsonl")
    prog.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(jobs), chunk_n):
        chunk_jobs = jobs[start : start + chunk_n]
        packed = []
        t_chunk = time.perf_counter()
        for job in chunk_jobs:
            ids, n_prompt = _teacher_force_ids(model, job)
            spec = SetEditSpec.from_state(job["edit"]) if job.get("edit") else None
            packed.append((job, ids, n_prompt, _pin_spec(spec, device)))
        with model.trace(temperature=0.0, top_p=1.0, max_tokens=1, ignore_eos=True) as tracer:
            saved = nnsight.save({})
            for i, (job, ids, n_prompt, spec) in enumerate(packed):
                with tracer.invoke(ids, temperature=0.0, top_p=1.0, max_tokens=1, ignore_eos=True):
                    edit_mode = str(job.get("edit_mode") or "clean")
                    if edit_mode in {"ablate", "keep"} and spec is not None:
                        out = _layer_out(model, layer, qwen35=qwen35)
                        _edit_generated_span(out, spec, n_prompt)
                    saved[f"lg_{i}"] = model.logits.save()
                    saved[i] = tracer.result.save()
        for i, (job, ids, n_prompt, spec) in enumerate(packed):
            logits = saved[f"lg_{i}"]
            vec = None
            if isinstance(logits, torch.Tensor):
                vec = logits.detach().float().reshape(-1, logits.shape[-1])[-1].cpu()
            row = {
                "question_id": job.get("question_id"),
                "method": job.get("method"),
                "n_prompt": n_prompt,
                "n_ids": len(ids),
                "margin": _score_vec(vec, job) if vec is not None else None,
                "n_captured": 1 if vec is not None else 0,
                "margin_error": None if vec is not None else "prefill logits empty",
            }
            out_rows.append(row)
            with prog.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "question_id": row["question_id"],
                            "method": row.get("method"),
                            "margin": row.get("margin"),
                        }
                    )
                    + "\n"
                )
        print(
            f"[worker] tf-prefill {start}:{start + len(chunk_jobs)}/{len(jobs)} "
            f"chunk_s={time.perf_counter() - t_chunk:.2f}",
            flush=True,
        )
    gen_s = time.perf_counter() - t_gen
    return {
        "ok": True,
        "phase": "generate",
        "mode": "teacher_force",
        "n": len(out_rows),
        "load_s": load_s,
        "gen_s": gen_s,
        "chunk": chunk_n,
        "rows": out_rows,
    }


def cmd_generate(args: argparse.Namespace) -> dict:
    import nnsight

    _tm4 = __import__("importlib").import_module("tasks.10_cross_model_steering.gemmascope_rows")
    apply_intervention_from_rows = _tm4.apply_intervention_from_rows

    jobs_path = Path(args.jobs)
    blob = pickle.loads(jobs_path.read_bytes())
    jobs = blob["jobs"]
    max_tokens = int(blob.get("max_new_tokens") or args.max_tokens)
    layer = _require_layer(args, blob)
    model_id = _require_model(args, blob)
    qwen35 = _is_qwen35(model_id)
    official = bool(blob.get("official", False))
    mode = str(blob.get("mode") or "steer")
    ignore_eos = (
        True
        if mode == "teacher_force"
        else ((not official) if blob.get("ignore_eos") is None else bool(blob["ignore_eos"]))
    )
    mode = str(blob.get("mode") or "steer")
    dump_resid_dir = blob.get("dump_resid_dir")
    parity_dir = blob.get("parity_dir")
    parity_steps = int(blob.get("parity_steps") or 0)
    chunk_n = int(blob.get("chunk") or 0) or max(1, min(16, len(jobs)))
    max_num_seqs = int(blob.get("max_num_seqs") or chunk_n)
    extra = None
    if mode == "teacher_force":
        extra = {
            "enable_prefix_caching": True,
            "max_num_batched_tokens": max(16384, int(max_num_seqs) * 512),
        }
    t_load = time.perf_counter()
    model = _build_engine(
        model_id=model_id,
        gpu_mem=args.gpu_mem,
        max_model_len=args.max_model_len,
        taps=bool(args.taps),
        layer=layer,
        max_num_seqs=max_num_seqs,
        extra=extra,
    )
    load_s = time.perf_counter() - t_load
    if mode == "teacher_force":
        return _teacher_force_trace(
            model, jobs, layer=layer, qwen35=qwen35, load_s=load_s, chunk_n=chunk_n
        )
    pair_rows = [_rows_from_state(j["rows"]) if j.get("rows") else None for j in jobs]
    edit_specs = []
    if mode in {"ablate", "keep", "teacher_force"}:
        _tm5 = __import__("importlib").import_module(
            "tasks.13_functional_concept_validation.sae_edit"
        )
        SetEditSpec = _tm5.SetEditSpec

        for job in jobs:
            edit_specs.append(SetEditSpec.from_state(job["edit"]) if job.get("edit") else None)
    else:
        edit_specs = [None] * len(jobs)
    sae_pack = None
    if mode == "keep" or any(j.get("edit_mode") == "keep" for j in jobs):
        sae_pack = _load_sae_pack(blob.get("sae_pack"))
    t_gen = time.perf_counter()
    out_rows = []
    n_tok = 0
    resid_dir = Path(dump_resid_dir) if dump_resid_dir else None
    pdir = Path(parity_dir) if parity_dir else None
    if resid_dir is not None:
        resid_dir.mkdir(parents=True, exist_ok=True)
    if pdir is not None:
        pdir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(jobs), chunk_n):
        chunk_jobs = jobs[start : start + chunk_n]
        chunk_rows = pair_rows[start : start + chunk_n]
        chunk_edits = edit_specs[start : start + chunk_n]
        keep_masks = None
        if mode == "keep" and sae_pack is not None:
            w_t = sae_pack.get("w_enc_t")
            d_sae = int(w_t.shape[1] if w_t is not None else sae_pack["w_enc"].shape[0])
            device = (w_t if w_t is not None else sae_pack["w_enc"]).device
            keep_masks = [
                _keep_mask(spec, job, d_sae, device) for spec, job in zip(chunk_edits, chunk_jobs)
            ]
        with model.trace(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        ) as tracer:
            saved = nnsight.save({})
            for i, (job, rows) in enumerate(zip(chunk_jobs, chunk_rows)):
                spec = chunk_edits[i]
                job_tokens = int(job.get("max_tokens") or max_tokens)
                with tracer.invoke(
                    job["prompt"],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=job_tokens,
                    ignore_eos=ignore_eos,
                ):
                    cap_x = []
                    cap_x2 = []
                    cap_lg = list().save()
                    saved[f"cap{i}"] = cap_lg
                    need_loop = (
                        (mode == "steer" and rows is not None)
                        or mode == "keep"
                        or (mode == "ablate" and spec is not None)
                        or mode == "teacher_force"
                        or (pdir is not None and rows is not None and parity_steps > 0)
                    )
                    if need_loop and mode == "teacher_force":
                        forced = list(job.get("forced_ids") or [])
                        import torch

                        for step in tracer.iter[:job_tokens]:
                            cap_lg.append(model.logits)
                            if int(step) < len(forced):
                                model.samples = torch.full_like(
                                    model.samples, int(forced[int(step)])
                                )
                            saved[i] = tracer.result.save()
                    elif need_loop:
                        for step in tracer.iter[:job_tokens]:
                            out = _layer_out(model, layer, qwen35=qwen35)
                            x, hidden, gemma = _unpack_layer_out(out)
                            need_cap = pdir is not None and step < max(parity_steps, 0)
                            if need_cap:
                                cap_x.append(x.save())
                            if mode == "steer" and rows is not None:
                                x2 = apply_intervention_from_rows(x, rows)
                                if need_cap:
                                    cap_x2.append(x2.save())
                                _writeback(hidden, x, x2, gemma)
                            elif mode == "keep" or (mode == "ablate" and spec is not None):
                                mask = keep_masks[i] if keep_masks is not None else None
                                x2 = _edit_hidden(x, spec, mode, sae_pack, job, keep_mask=mask)
                                _writeback(hidden, x, x2, gemma)
                            saved[i] = tracer.result.save()
                            if pdir is not None:
                                saved[f"cx_{i}"] = cap_x
                                saved[f"c2_{i}"] = cap_x2
                    else:
                        saved[i] = tracer.result.save()
                    if pdir is not None:
                        saved[f"cx_{i}"] = cap_x
                        saved[f"c2_{i}"] = cap_x2
        print(
            f"[worker] chunk {start}:{start + len(chunk_jobs)} keys={list(saved.keys())}",
            flush=True,
        )
        Path("/workspace/SAEReasoning/experiment/tmp/13_worker_keys.txt").write_text(
            repr(list(saved.keys())) + "\n", encoding="utf-8"
        )
        for i, job in enumerate(chunk_jobs):
            res = saved[i]
            text = res.outputs[0].text
            n = len(res.outputs[0].token_ids)
            n_tok += n
            row = {
                "question_id": job.get("question_id"),
                "method": job.get("method"),
                "feature_i": job.get("feature_i"),
                "feature_j": job.get("feature_j"),
                "n_gen_tokens": n,
                "gen_text": text,
                "prompt": job.get("prompt"),
                "token_ids": [int(t) for t in list(res.outputs[0].token_ids)],
            }
            if mode == "teacher_force":
                try:
                    cap = saved[f"cap{i}"]
                except Exception:
                    cap = None
                info = _margin_from_saved(cap, i, job)
                row["margin"] = info.get("margin")
                row["margin_error"] = info.get("margin_error")
                row["margin_step_hit"] = info.get("margin_step_hit")
                row["n_captured"] = info.get("n_captured")
            if resid_dir is not None and mode == "baseline":
                qid_safe = (
                    str(job.get("question_id"))
                    .replace("/", "__")
                    .replace("\\", "__")
                    .replace(" ", "_")
                )
                resid_path = _dump_generated_resid(
                    model,
                    prompt=str(job["prompt"]),
                    text=text,
                    layer=layer,
                    dest=resid_dir / f"{qid_safe}.pt",
                    qwen35=qwen35,
                    n_gen=n,
                )
                row["resid_path"] = str(resid_path) if resid_path else None
            if pdir is not None:
                try:
                    cx = saved[f"cx_{i}"]
                    c2 = saved[f"c2_{i}"]
                except Exception:
                    cx, c2 = [], []
                tag = job.get("parity_tag") or job.get("question_id")
                _save_parity_caps(
                    pdir / f"parity_{tag}.pt",
                    question_id=job.get("question_id"),
                    cap_x=cx or [],
                    cap_x2=c2 or [],
                )
            out_rows.append(row)
            stream_path = blob.get("stream_path")
            if stream_path:
                with Path(stream_path).open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
    gen_s = time.perf_counter() - t_gen
    return {
        "ok": True,
        "phase": "generate",
        "mode": mode,
        "n": len(out_rows),
        "load_s": load_s,
        "gen_s": gen_s,
        "n_gen_tokens": n_tok,
        "tok_per_s": n_tok / max(gen_s, 1e-6),
        "gens_per_min": len(out_rows) / max(gen_s, 1e-6) * 60.0,
        "taps": bool(args.taps),
        "chunk": chunk_n,
        "official": official,
        "ignore_eos": ignore_eos,
        "model_id": model_id,
        "layer": layer,
        "temperature": 0.0,
        "top_p": 1.0,
        "rows": out_rows,
    }


def _save_parity_caps(dest, *, question_id, cap_x, cap_x2):
    import torch

    def _last(t):
        if t.ndim == 3:
            return t[0, -1]
        if t.ndim == 2:
            return t[-1]
        return t.reshape(-1)

    xs = []
    x2s = []
    dtypes = []
    changed = []
    for a, b in zip(cap_x, cap_x2):
        ta = a.detach() if hasattr(a, "detach") else torch.as_tensor(a)
        tb = b.detach() if hasattr(b, "detach") else torch.as_tensor(b)
        ta = _last(ta)
        tb = _last(tb)
        dtypes.append(str(ta.dtype))
        changed.append(float((tb != ta).to(dtype=torch.float32).mean().item()))
        xs.append(ta.detach().float().cpu().contiguous().reshape(-1))
        x2s.append(tb.detach().float().cpu().contiguous().reshape(-1))
    if not xs or not x2s:
        print(f"[worker] parity dump empty {dest}", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "question_id": question_id,
            "x": torch.stack(xs, 0),
            "x2": torch.stack(x2s, 0),
            "dtype": dtypes[0],
            "changed_frac": changed,
        },
        dest,
    )


def _dump_generated_resid(
    model, *, prompt: str, text: str, layer: int, dest: Path, n_gen: int = 0, qwen35: bool = False
):
    import torch

    full = str(prompt) + str(text)
    try:
        with model.trace(
            full,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            ignore_eos=True,
        ) as tracer:
            out = _layer_out(model, layer, qwen35=qwen35).save()
            _ = tracer.result.save()
        tens = _unpack_full_sequence(out).cpu()
        n_keep = int(n_gen) if int(n_gen) > 0 else 0
        if tens.ndim == 2 and n_keep and n_keep < tens.shape[0]:
            tens = tens[-n_keep:]
        elif tens.ndim == 2:
            tok = getattr(model, "tokenizer", None)
            plen = 0
            if tok is not None:
                plen = len(tok.encode(prompt, add_special_tokens=False))
            if plen < tens.shape[0]:
                tens = tens[plen:]
        if tens.ndim == 1:
            tens = tens.unsqueeze(0)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tens.contiguous().half(), dest)
        return dest
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] resid dump failed {dest.name}: {exc}", flush=True)
        return None


def cmd_probe_layer(args: argparse.Namespace) -> dict:
    """Print layer output type for tap selection (E0)."""
    model_id = _require_model(args)
    layer = _require_layer(args)
    model = _build_engine(
        model_id=model_id,
        gpu_mem=args.gpu_mem,
        max_model_len=min(int(args.max_model_len), 2048),
        taps=True,
        layer=layer,
        max_num_seqs=1,
    )
    prompt = "Hello"
    with model.trace(prompt, temperature=0.0, top_p=1.0, max_tokens=1, ignore_eos=True) as tracer:
        out = _layer_out(model, layer, qwen35=_is_qwen35(model_id)).save()
        _ = tracer.result.save()
    info = {"type": type(out).__name__}
    if isinstance(out, (tuple, list)):
        info["n"] = len(out)
        info["elem0"] = str(getattr(out[0], "shape", None))
        if len(out) > 1:
            info["elem1"] = str(getattr(out[1], "shape", None))
    else:
        info["shape"] = str(getattr(out, "shape", None))
    return {"ok": True, "phase": "probe", "model_id": model_id, "layer": layer, "output": info}


def cmd_dump_resid(args: argparse.Namespace) -> dict:
    """Dump layer resid for existing prompt+gen_text (no new sampling)."""
    import torch

    jobs_path = Path(args.jobs)
    blob = pickle.loads(jobs_path.read_bytes())
    jobs = blob["jobs"]
    layer = _require_layer(args, blob)
    model_id = _require_model(args, blob)
    dump_resid_dir = Path(blob.get("dump_resid_dir") or "")
    if not dump_resid_dir:
        return {"ok": False, "error": "dump_resid_dir required"}
    dump_resid_dir.mkdir(parents=True, exist_ok=True)
    chunk_n = max(1, int(blob.get("chunk") or 16))
    t_load = time.perf_counter()
    model = _build_engine(
        model_id=model_id,
        gpu_mem=args.gpu_mem,
        max_model_len=args.max_model_len,
        taps=True,
        layer=layer,
        max_num_seqs=chunk_n,
        extra={
            "enable_prefix_caching": True,
            "max_num_batched_tokens": max(16384, int(chunk_n) * 512),
        },
    )
    load_s = time.perf_counter() - t_load
    t0 = time.perf_counter()
    rows = []
    n_ok = 0
    pending = []
    for job in jobs:
        qid = job.get("question_id")
        qid_safe = str(qid).replace("/", "__").replace("\\", "__").replace(" ", "_")
        dest = dump_resid_dir / f"{qid_safe}.pt"
        if dest.is_file() and not bool(blob.get("force")):
            rows.append({"question_id": qid, "resid_path": str(dest), "skipped": True})
            n_ok += 1
            continue
        pending.append((job, dest))
    tok = getattr(model, "tokenizer", None)
    qwen35 = _is_qwen35(model_id)
    for start in range(0, len(pending), chunk_n):
        chunk = pending[start : start + chunk_n]
        packed = []
        for job, dest in chunk:
            prompt = str(job.get("prompt") or "")
            text = str(job.get("gen_text") or job.get("text") or "")
            if job.get("input_ids"):
                ids = [int(x) for x in job["input_ids"]]
                n_prompt = int(job.get("n_prompt") or 0)
            elif tok is not None:
                prompt_ids = list(tok.encode(prompt, add_special_tokens=False))
                gen_ids = list(tok.encode(text, add_special_tokens=False)) if text else []
                ids = prompt_ids + gen_ids
                n_prompt = len(prompt_ids)
            else:
                ids, n_prompt = [], 0
            packed.append((job, dest, ids, n_prompt))
        # keep generated tail if the concatenated sequence exceeds the engine cap
        max_ids = max(8, int(args.max_model_len) - 8)
        trimmed = []
        for job, dest, ids, n_prompt in packed:
            if not ids:
                print(f"[worker] dump skip empty ids q={job.get('question_id')}", flush=True)
                continue
            if len(ids) > max_ids:
                cut = len(ids) - max_ids
                ids = ids[-max_ids:]
                n_prompt = max(0, int(n_prompt) - cut)
            trimmed.append((job, dest, ids, n_prompt))
        packed = trimmed
        if not packed:
            continue
        try:
            # Same slice as 10a: one forward of prompt+text, keep the last n_gen
            # positions. Do not encode the prompt and the generation separately.
            for job, dest, ids, n_prompt in packed:
                n_gen = int(
                    job.get("n_gen_tokens")
                    or job.get("generated_tokens")
                    or max(0, len(ids) - int(n_prompt))
                )
                path = _dump_generated_resid(
                    model,
                    prompt=str(job.get("prompt") or ""),
                    text=str(job.get("gen_text") or job.get("text") or ""),
                    layer=layer,
                    dest=dest,
                    n_gen=n_gen,
                    qwen35=qwen35,
                )
                ok = path is not None
                n_ok += int(ok)
                n_tok = 0
                if ok:
                    import torch

                    saved_t = torch.load(path, map_location="cpu", weights_only=False)
                    n_tok = int(saved_t.shape[0]) if hasattr(saved_t, "shape") else 0
                rows.append(
                    {
                        "question_id": job.get("question_id"),
                        "resid_path": str(path) if path else None,
                        "ok": ok,
                        "n_tok": n_tok,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] dump batch {start} failed ({exc}); serial fallback", flush=True)
            for job, dest, ids, n_prompt in packed:
                path = _dump_generated_resid(
                    model,
                    prompt=str(job.get("prompt") or ""),
                    text=str(job.get("gen_text") or job.get("text") or ""),
                    layer=layer,
                    dest=dest,
                    n_gen=int(
                        job.get("n_gen_tokens")
                        or job.get("generated_tokens")
                        or max(0, len(ids) - n_prompt)
                    ),
                    qwen35=qwen35,
                )
                ok = path is not None
                n_ok += int(ok)
                rows.append(
                    {
                        "question_id": job.get("question_id"),
                        "resid_path": str(path) if path else None,
                        "ok": ok,
                    }
                )
        print(f"[worker] dump {start + len(chunk)}/{len(pending)}", flush=True)
    return {
        "ok": n_ok == len(jobs),
        "phase": "dump_resid",
        "n": len(jobs),
        "n_ok": n_ok,
        "load_s": load_s,
        "wall_s": time.perf_counter() - t0,
        "chunk": chunk_n,
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["smoke", "generate", "probe", "dump_resid"])
    p.add_argument("--jobs", default="")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--taps", action="store_true")
    p.add_argument("--model-id", default="")
    p.add_argument("--layer", type=int, default=None)
    args = p.parse_args()
    try:
        if args.cmd == "smoke":
            out = cmd_smoke(args)
        elif args.cmd == "probe":
            out = cmd_probe_layer(args)
        elif args.cmd == "dump_resid":
            if not args.jobs:
                raise ValueError("--jobs required for dump_resid")
            out = cmd_dump_resid(args)
        else:
            if not args.jobs:
                raise ValueError("--jobs required for generate")
            out = cmd_generate(args)
        _emit(out)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _emit(
            {
                "ok": False,
                "phase": args.cmd,
                "error": type(exc).__name__ + ": " + str(exc),
            }
        )
        raise SystemExit(1)


if __name__ == "__main__":
    from tasks.generate.backend.settings import apply_process_env

    apply_process_env()
    main()
