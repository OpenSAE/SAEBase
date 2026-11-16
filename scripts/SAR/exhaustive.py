"""Measure every Top10 coalition with the original SAEBase scoring protocol.

Run from the repository root: python -m scripts.SAR.exhaustive
The lower 22 blocks are evaluated once per question, on CUDA with CPU weight
offload. Every coalition uses the full-context final ten blocks, without a KV
cache or altered batch shape. Replay gates must pass before old scores are reused.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from saebase.api import FeatureSnapshot, edit_features, load_registered
from saebase.seed import apply_seed
from tasks.capture.answer import ContextTailScorer, answer_patch
from tasks.sae_runtime import pair_state_fingerprints

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
PROTOCOL = "sar.top10.exhaustive.full_context_tail.v1"


def write_json(path, document):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def log(event, **values):
    print(json.dumps({"time": stamp(), "event": event, **values}, ensure_ascii=False), flush=True)


class TailScorer:
    """Keep native arithmetic on CUDA, offloading only unused model weights."""

    def __init__(self, pair):
        self.pair = pair
        self.model = pair.model.model
        if pair.model.backend != "transformer_lens":
            raise ValueError("Historical scores require TransformerLens")
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.cfg.device = "cuda"
        self.model.cfg.n_devices = 1
        # TransformerLens transposes and materializes this 1 GB matrix at every
        # call. Store the identical values in its eventual linear-layer layout.
        weight = self.model.unembed.W_U
        weight.data = weight.data.T.contiguous().T
        for block in self.model.blocks[22:]:
            block.to("cuda")
        self.model.ln_final.to("cuda")
        pair.sae.to("cuda")
        for module in [self.model.embed, *self.model.blocks[:22]]:
            module.register_forward_pre_hook(self.before)
            module.register_forward_hook(self.after)
        self.model.unembed.to("cuda")

    @staticmethod
    def before(module, args):
        module.to("cuda")

    @staticmethod
    def after(module, args, output):
        module.to("cpu")
        return output

    def prepare(self, arrays):
        self.prompt = torch.as_tensor(arrays["source_tokens"], device="cuda")
        self.answer = torch.as_tensor(arrays["answer_token_ids"], device="cuda")[None]
        self.position = self.prompt.shape[1] - 1
        tokens = torch.cat([self.prompt, self.answer[:, :-1]], dim=1)
        # Release the large output projection while streaming the prefix.
        self.model.unembed.to("cpu")
        torch.cuda.empty_cache()
        self.hidden = self.model(tokens, stop_at_layer=22).detach()
        torch.cuda.empty_cache()
        self.model.unembed.to("cuda")
        reference = torch.as_tensor(arrays["readout_activation"], device="cuda")
        error = float((self.hidden[0, self.position].float() - reference).abs().max())
        if error != 0:
            raise RuntimeError(f"Historical answer-context activation differs: {error}")
        self.tail_scorer = ContextTailScorer(
            self.model, self.hidden, self.answer, self.prompt.shape[1], 22
        )
        return error

    def native_target(self, snapshot, feature_ids, state):
        remove = {int(fid): 0.0 for bit, fid in enumerate(feature_ids) if not state & (1 << bit)}
        patch = edit_features(self.pair.sae, snapshot, replacements=remove)
        return answer_patch(snapshot.activation, self.hidden[0, self.position], patch.activation)

    def prepare_edits(self, snapshot):
        if self.pair.sae.cfg.normalize_activations != "none":
            raise ValueError("Cached native decoding is scoped to the folded Llama SAE")
        self.decoded_native = self.pair.sae.decode(snapshot.codes[None])[0].detach()

    def target(self, snapshot, feature_ids, state):
        # Same two native decoder calls and BF16 rounding as edit_features;
        # only the unchanged first decoder output is cached. The known finite
        # zero replacements can be assigned together without scalar GPU reads.
        absent = [int(fid) for bit, fid in enumerate(feature_ids) if not state & (1 << bit)]
        if absent:
            edited_codes = snapshot.codes.clone()
            edited_codes[absent] = 0.0
            after = self.pair.sae.decode(edited_codes[None])[0]
            change = (after - self.decoded_native).to(snapshot.activation.dtype)
            edited_activation = snapshot.activation + change
        else:
            edited_activation = snapshot.activation
        return answer_patch(snapshot.activation, self.hidden[0, self.position], edited_activation)

    def score(self, target):
        return self.tail_scorer(target)

    def oracle(self, target):
        values = self.pair.model.answer_token_logprobs(
            prompt_tokens=self.prompt,
            answer_tokens=self.answer,
            site=self.pair.site,
            activation=target,
        )
        return float(values.double().mean())


def source_arrays(question):
    local = OUTPUT / question["key"] / "source_sar.npz"
    path = local if local.exists() else Path(question["artifact"]["absolute_path"])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != question["artifact"]["sha256"]:
        raise ValueError(f"Source checksum mismatch: {path}")
    if not local.exists():
        local.parent.mkdir(exist_ok=True)
        local.write_bytes(content)
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def original_scores(arrays):
    result = {}
    powers = 1 << np.arange(len(arrays["feature_ids"]))
    for mask, response in zip(arrays["masks"], arrays["responses"]):
        state = int(np.dot(mask, powers))
        if state in result and abs(result[state] - float(response)) > 1e-10:
            raise ValueError("Repeated historical coalition has inconsistent scores")
        result.setdefault(state, float(response))
    result[(1 << len(powers)) - 1] = float(arrays["full_response"])
    return result


def measure_question(pair, scorer, question):
    arrays = source_arrays(question)
    ids = arrays["feature_ids"].tolist()
    codes = torch.zeros(question["feature_count"], dtype=torch.float32, device="cuda")
    for feature in question["active_features"]:
        codes[feature["feature_id"]] = feature["activation"]
    if not torch.equal(codes[ids].cpu(), torch.from_numpy(arrays["selected_codes"])):
        raise ValueError("Full native code vector differs from historical selected codes")
    # NPZ export promoted the native BF16 activation to FP32. Restore its
    # original dtype before edit_features, which rounds the decoder delta and
    # edited activation in that dtype as in the original Capture execution.
    activation = torch.as_tensor(arrays["activation"], device="cuda").to(scorer.model.cfg.dtype)
    snapshot = FeatureSnapshot(
        tokens=torch.as_tensor(arrays["source_tokens"], device="cuda"),
        site=pair.site,
        activation=activation,
        codes=codes,
        reconstruction=torch.zeros_like(activation),
    )
    directory = OUTPUT / question["key"]
    directory.mkdir(exist_ok=True)
    source = original_scores(arrays)
    states = sorted(source)
    activation_error = scorer.prepare(arrays)
    # Verify the actual native encoding, not only the saved selected coordinates.
    encoded = pair.sae.encode(activation.to(next(pair.sae.parameters()).dtype)[None])[0]
    code_error = float((encoded - codes).abs().max())
    if code_error != 0:
        raise RuntimeError(f"Native SAE code replay differs: {code_error}")
    scorer.prepare_edits(snapshot)
    checks = []
    probes = sorted(set([states[0], states[len(states) // 2], states[-1]]))
    for state in probes:
        target = scorer.target(snapshot, ids, state)
        native = scorer.native_target(snapshot, ids, state)
        if not torch.equal(target, native):
            raise RuntimeError("Cached decoder patch differs from native edit_features")
        score = scorer.score(target)
        # One complete native forward per question establishes tail parity;
        # the other probes independently replay two more historical scores.
        oracle = scorer.oracle(target) if state == probes[0] else None
        check = {"state": state, "tail": score, "full_forward": oracle, "historical": source[state]}
        checks.append(check)
        log("parity", question=question["key"], **check)
        if (oracle is not None and abs(score - oracle) > 1e-7) or abs(score - source[state]) > 1e-7:
            raise RuntimeError(f"Historical/full-context tail replay gate failed: {check}")
    provenance = {
        "protocol": PROTOCOL,
        "question": question["key"],
        "source_sha256": question["artifact"]["sha256"],
        "candidate_ids": ids,
        "activation_max_abs_error": activation_error,
        "codes_max_abs_error": code_error,
        "native_edit_patch_max_abs_error": 0.0,
        "parity_checks": checks,
        "created_at": stamp(),
    }
    write_json(directory / "verification.json", provenance)
    progress = directory / "measurements.jsonl"
    measured = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            item = json.loads(line)
            if item["protocol"] != PROTOCOL or item["source_sha256"] != provenance["source_sha256"]:
                raise ValueError("Resume provenance mismatch")
            measured[item["state"]] = item["g"]
    scores = dict(source)
    scores.update(measured)
    started = time.monotonic()
    with progress.open("a", buffering=1) as stream:
        for state in range(1 << len(ids)):
            if state in scores:
                continue
            value = scorer.score(scorer.target(snapshot, ids, state))
            if not np.isfinite(value) or value > 0:
                raise RuntimeError("Invalid answer mean log probability")
            scores[state] = value
            stream.write(
                json.dumps(
                    {
                        "state": state,
                        "g": value,
                        "protocol": PROTOCOL,
                        "source_sha256": provenance["source_sha256"],
                    }
                )
                + "\n"
            )
            if len(scores) % 50 == 0:
                log(
                    "progress",
                    question=question["key"],
                    states=len(scores),
                    seconds=time.monotonic() - started,
                )
    all_states = np.arange(1 << len(ids))
    masks = ((all_states[:, None] >> np.arange(len(ids))) & 1).astype(bool)
    temporary = directory / "coalitions.tmp.npz"
    np.savez_compressed(
        temporary,
        feature_ids=ids,
        masks=masks,
        responses=np.array([scores[int(state)] for state in all_states]),
        historical=np.array([int(state) in source for state in all_states]),
        answer_token_ids=arrays["answer_token_ids"],
    )
    temporary.replace(directory / "coalitions.npz")
    write_json(
        directory / "complete.json",
        {
            **provenance,
            "completed_at": stamp(),
            "states": len(scores),
            "historical_states": len(source),
            "new_states": len(scores) - len(source),
            "seconds": time.monotonic() - started,
            "coalitions_sha256": hashlib.sha256(
                (directory / "coalitions.npz").read_bytes()
            ).hexdigest(),
        },
    )
    log("complete", question=question["key"], seconds=time.monotonic() - started)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    apply_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.27)
    plan = json.loads((OUTPUT / "plan.json").read_text())
    questions = plan["questions"][: args.limit]
    pending = [q for q in questions if not (OUTPUT / q["key"] / "complete.json").exists()]
    if not pending:
        log("all_complete", questions=len(questions))
        return
    log("loading", pending=len(pending))
    pair = load_registered(
        model_key=plan["model"], sae_key=plan["sae"], layer=plan["layer"], device="cpu"
    )
    # Native activation-norm folding is device-sensitive in FP32. Load the SAE
    # through the original CUDA path even while model weights begin on CPU.
    from saebase.api.sae_loading import load_sae_for_profile

    native_sae, _ = load_sae_for_profile(plan["sae"], plan["layer"], device="cuda")
    pair = replace(pair, sae=native_sae)
    model_sha, sae_sha = pair_state_fingerprints(pair)
    expected = pending[0]["native_input_identity"]
    if model_sha != expected["model_sha256"] or sae_sha != expected["sae_sha256"]:
        raise RuntimeError(f"Checkpoint fingerprint mismatch: {model_sha}, {sae_sha}")
    write_json(
        OUTPUT / "execution.json",
        {
            "protocol": PROTOCOL,
            "started_at": stamp(),
            "model_sha256": model_sha,
            "sae_sha256": sae_sha,
            "gpu_memory_fraction": 0.27,
            "arithmetic": "native BF16 model; FP32 SAE; FP32 log_softmax; FP64 mean",
            "scope": "all masks of each original filtered Top10; all backgrounds",
            "scorer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    scorer = TailScorer(pair)
    for question in pending:
        log("question", key=question["key"])
        measure_question(pair, scorer, question)
        from scripts.SAR.build_exhaustive_report import main as build_report

        build_report()
    log("all_complete", questions=len(questions))


if __name__ == "__main__":
    main()
