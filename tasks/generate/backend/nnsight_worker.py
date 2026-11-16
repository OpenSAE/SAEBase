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






















def cmd_generate(args: argparse.Namespace) -> dict:
    """Run baseline generation; feature edits use the current generate/intervene task APIs."""
    import nnsight
    blob = pickle.loads(Path(args.jobs).read_bytes())
    mode = str(blob.get("mode") or "baseline")
    jobs = blob["jobs"]
    if mode != "baseline" or any(job.get("rows") or job.get("edit") for job in jobs):
        raise ValueError("This worker supports baseline generation only. Use tasks.generate for supported feature interventions.")
    model_id, layer = _require_model(args, blob), _require_layer(args, blob)
    max_tokens = int(blob.get("max_new_tokens") or args.max_tokens)
    chunk_n = max(1, int(blob.get("chunk") or min(16, max(1, len(jobs)))))
    official = bool(blob.get("official", False))
    ignore_eos = not official if blob.get("ignore_eos") is None else bool(blob["ignore_eos"])
    started = time.perf_counter()
    model = _build_engine(model_id=model_id, gpu_mem=args.gpu_mem,
                          max_model_len=args.max_model_len, taps=bool(args.taps),
                          layer=layer, max_num_seqs=chunk_n)
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    result_rows, token_count = [], 0
    for start in range(0, len(jobs), chunk_n):
        chunk = jobs[start:start + chunk_n]
        with model.trace(temperature=0.0, top_p=1.0, max_tokens=max_tokens, ignore_eos=ignore_eos) as tracer:
            saved = nnsight.save({})
            for index, job in enumerate(chunk):
                with tracer.invoke(job["prompt"], temperature=0.0, top_p=1.0,
                                   max_tokens=int(job.get("max_tokens") or max_tokens), ignore_eos=ignore_eos):
                    saved[index] = tracer.result.save()
        for index, job in enumerate(chunk):
            output = saved[index].outputs[0]
            ids = [int(value) for value in output.token_ids]
            token_count += len(ids)
            row = {"question_id": job.get("question_id"), "method": job.get("method"),
                   "n_gen_tokens": len(ids), "gen_text": output.text,
                   "prompt": job["prompt"], "token_ids": ids}
            if blob.get("dump_resid_dir"):
                name = str(job.get("question_id")).replace("/", "__").replace(chr(92), "__").replace(" ", "_")
                dest = Path(blob["dump_resid_dir"]) / (name + ".pt")
                path = _dump_generated_resid(model, prompt=str(job["prompt"]), text=output.text,
                                              layer=layer, dest=dest, n_gen=len(ids), qwen35=_is_qwen35(model_id))
                row["resid_path"] = str(path) if path else None
            result_rows.append(row)
            if blob.get("stream_path"):
                with Path(blob["stream_path"]).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, default=str) + "\n")
    elapsed = time.perf_counter() - started
    return {"ok": True, "phase": "generate", "mode": "baseline", "model_id": model_id,
            "layer": layer, "n": len(result_rows), "rows": result_rows,
            "n_gen_tokens": token_count, "load_s": load_s, "gen_s": elapsed,
            "tok_per_s": token_count / max(elapsed, 1e-6), "chunk": chunk_n,
            "temperature": 0.0, "top_p": 1.0, "official": official, "ignore_eos": ignore_eos}





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
    """Inspect the layer output shape for the selected tap."""
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
