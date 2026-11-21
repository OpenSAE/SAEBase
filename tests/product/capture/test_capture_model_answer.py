"""Original Capture answers use the exact loaded model/input and survive reuse."""

from types import SimpleNamespace
import hashlib
import json

from omegaconf import OmegaConf
import pytest
import torch

from saebase.results import completion_summary
from tasks.capture import model_answer as owner


class Tokenizer:
    padding_side = "left"
    eos_token_id = 15
    pad_token_id = 0
    bos_token_id = 1
    special_tokens_map = {"bos_token": "BOS", "eos_token": "EOS"}

    def encode(self, text, add_special_tokens=False):
        return [int(token) for token in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if not skip_special_tokens or i not in (0, 1, 15))

    def get_vocab(self):
        return {str(i): i for i in range(16)}


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create(
        {
            "task": {"capture": {"model_answer": {"max_new_tokens": 3, "batch_size": 8}}},
            "paths": {"root": str(tmp_path)},
            "seed": 42,
        }
    )


@pytest.fixture
def pair():
    # Real upstream HuggingFace generation; tiny random model, no downloads.
    from transformers import GPT2Config, GPT2LMHeadModel
    from models.hf_hooks_backend import HFHooksModel
    from saebase.api import ModelAdapter

    torch.manual_seed(17)
    raw = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=32,
            bos_token_id=1,
            eos_token_id=15,
        )
    ).eval()
    adapter = ModelAdapter(HFHooksModel(raw, Tokenizer(), model_name="tiny", device="cpu"))
    return SimpleNamespace(model=adapter, sae=torch.nn.Linear(4, 4))


def test_upstream_generation_exact_inputs_and_cross_mode_cache(cfg, pair, monkeypatch):
    raw = pair.model.model._model
    seen = []
    original = raw.generate

    def generate(ids, **kwargs):
        seen.append(ids.clone())
        assert kwargs["do_sample"] is False and kwargs["use_cache"] is True
        return original(ids, **kwargs)

    monkeypatch.setattr(raw, "generate", generate)
    before = {k: v.clone() for k, v in raw.state_dict().items()}
    result = owner.answers(cfg, pair, ["2 3", "4 5"])
    assert all(r["status"] == "completed" for r in result)
    assert seen[0].tolist() == [[1, 2, 3], [1, 4, 5]]
    expected = original(seen[0][:1], max_new_tokens=3, do_sample=False, pad_token_id=0)
    assert result[0]["token_ids"] == expected[0, 3:].tolist()
    cfg.task.mode = "I"
    reused = owner.answers(cfg, pair, ["4 5", "2 3"])
    assert len(seen) == 1 and all(r["reused"] for r in reused)
    assert reused[1]["text"] == result[0]["text"]
    assert all(torch.equal(v, before[k]) for k, v in raw.state_dict().items())
    with torch.no_grad():
        next(raw.parameters()).add_(0.01)
    owner.answers(cfg, pair, ["2 3"])
    assert len(seen) == 2  # Checkpoint changes invalidate answers.


def test_context_limit_and_generation_failure_do_not_drop_features(cfg, pair, monkeypatch):
    pair.model.model.cfg.n_ctx = 3
    blocked = owner.answers(cfg, pair, ["2 3"])[0]
    assert blocked["status"] == "blocked" and blocked["finish_reason"] == "context_limit"
    result = owner.attach(
        {
            "top_features": [{"feature_id": 2}],
            "completion": completion_summary([{"status": "completed"}]),
        },
        blocked,
    )
    assert result["top_features"] == [{"feature_id": 2}]
    assert result["completion"]["status"] == "incomplete"
    assert result["stages"]["features_and_annotations"]["status"] == "complete"
    pair.model.model.cfg.n_ctx = 32

    def fail(*args, **kwargs):
        raise RuntimeError("upstream failed")

    monkeypatch.setattr(pair.model.model._model, "generate", fail)
    failure = owner.answers(cfg, pair, ["2 3"])[0]
    assert failure["status"] == "failed" and failure["error"]["message"] == "upstream failed"
    fixed = owner.attach(result, {"status": "completed", "text": "answer"})
    assert fixed["completion"]["status"] == "complete"


@pytest.mark.parametrize("scope", ["last_token", "all_tokens"])
@pytest.mark.parametrize("mode", ["non_whiten", "whiten", "g_s", "I", "sar"])
def test_common_capture_path_attaches_answer(cfg, monkeypatch, scope, mode):
    from tasks.capture import operation

    cfg.task.output_scope = scope
    cfg.task.mode = mode
    prompt = "original prompt, without gold answer"
    sentinel = SimpleNamespace()
    monkeypatch.setattr(
        operation,
        "_capture_features",
        lambda *a, **kw: {
            "top_features": [{"feature_id": 7}],
            "completion": completion_summary([{"status": "completed"}]),
        },
    )
    calls = []

    def answer(cfg, pair, prompts):
        calls.append((pair, prompts))
        return [{"status": "completed", "text": "model continuation"}]

    monkeypatch.setattr(owner, "answers", answer)
    result = operation._capture(cfg, pair=sentinel, prompt=prompt)
    assert result["model_answer"]["text"] == "model continuation"
    assert calls == [(sentinel, [prompt])]
    operation._capture(cfg, pair=sentinel, prompt=prompt, model_answer=result["model_answer"])
    assert len(calls) == 1


def test_old_ledger_answer_supplement_and_reader(cfg, pair, tmp_path, monkeypatch):
    from saebase.store import sha256_state_dict
    from tasks.capture.annotations import read_samples
    from tasks.capture.task import CaptureTask
    import saebase.datasets.selection

    cfg.model = {"name": "tiny"}
    cfg.dataset = {"name": "synthetic", "sample_count": 1}
    row = {
        "sample_id": "q",
        "status": "completed",
        "result": {
            "top_features": [],
            "active_features": [],
            "input_identity": {"model_sha256": sha256_state_dict(pair.model.model)},
            "prompt_sha256": hashlib.sha256(b"2 3").hexdigest(),
            "completion": completion_summary([{"status": "completed"}]),
        },
    }
    (tmp_path / "reports").mkdir()
    (tmp_path / "metadata").mkdir()
    ledger = tmp_path / "reports/capture_samples.jsonl"
    ledger.write_text(json.dumps(row) + "\n")
    before = ledger.read_bytes()
    manifest = {
        "task": "capture",
        "configuration": OmegaConf.to_container(cfg),
        "result": {"summary": {"samples_path": str(ledger)}, "status": "executed", "ok": True},
    }
    (tmp_path / "metadata/manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        saebase.datasets.selection,
        "read_selection",
        lambda *a: [{"sample_id": "q", "prompt": "2 3"}],
    )
    update = owner.refresh(tmp_path, pair=pair)
    assert update["model_answer"]["status"] == "complete"
    assert ledger.read_bytes() == before
    result = list(read_samples(tmp_path))[0]
    assert result["result"]["model_answer"]["status"] == "completed"
    assert result["status"] == "completed"
    assert (
        CaptureTask().read_result(tmp_path, manifest)["result"]["summary"]["stages"][
            "model_answer"
        ]["status"]
        == "complete"
    )


def test_stream_failure_keeps_already_observed_rows(cfg, monkeypatch):
    def source():
        yield {"prompt": "a"}
        yield {"prompt": "b"}
        raise RuntimeError("source interrupted")

    monkeypatch.setattr(
        owner,
        "answers",
        lambda cfg, pair, prompts: [{"status": "completed", "text": p} for p in prompts],
    )
    iterator = owner.answer_rows(cfg, None, source())
    assert next(iterator)[0]["prompt"] == "a"
    assert next(iterator)[0]["prompt"] == "b"
    with pytest.raises(RuntimeError, match="source interrupted"):
        next(iterator)


@pytest.mark.parametrize("value", [0, -1, True, "8"])
def test_invalid_generation_limits_fail_before_loading(cfg, value):
    cfg.task.capture.model_answer.max_new_tokens = value
    with pytest.raises(ValueError, match="model_answer.max_new_tokens"):
        owner.settings(cfg)


def test_transformerlens_upstream_generation(cfg):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from saebase.api import ModelAdapter

    raw = HookedTransformer(
        HookedTransformerConfig(
            n_layers=1,
            d_model=8,
            n_heads=1,
            d_head=8,
            d_mlp=16,
            d_vocab=16,
            n_ctx=32,
            act_fn="relu",
            device="cpu",
            seed=17,
        )
    )
    raw.tokenizer = Tokenizer()
    raw.to_tokens = lambda prompt: torch.tensor([[1, *raw.tokenizer.encode(prompt)]])
    raw.eval()
    pair = SimpleNamespace(model=ModelAdapter(raw), sae=torch.nn.Linear(4, 4))
    result = owner.answers(cfg, pair, ["2 3", "4 5", "2 3"])
    assert all(r["status"] == "completed" for r in result), result
    assert result[0]["token_ids"] == result[2]["token_ids"] and result[2]["reused"]
    expected = raw.generate(
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=3,
        do_sample=False,
        eos_token_id=[15],
        return_type="tokens",
        verbose=False,
    )
    assert result[0]["token_ids"] == expected[0, 3:].tolist()


def test_real_capture_dispatch_contains_answer_and_preserves_features(
    cfg, pair, monkeypatch, tmp_path
):
    from saebase.api import SAEIdentity
    from saebase.results import EvaluationResult
    from tasks.capture import operation
    import saebase.datasets.selection as datasets

    class SAE(torch.nn.Module):
        cfg = SimpleNamespace(d_in=8, d_sae=4, normalize_activations=None)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(8, 4))

        def encode(self, hidden):
            return (hidden @ self.weight).relu()

        def decode(self, codes):
            return codes @ self.weight.T

    pair.sae = SAE().eval()
    pair.identity = SAEIdentity(
        "tiny", "tiny", "tiny", "synthetic", "blocks.0.hook_resid_post", 0, "synthetic", 8, 4
    )
    pair.site = pair.identity.site
    cfg.model = {"name": "tiny"}
    cfg.dataset = {"name": "synthetic", "sample_count": 2}
    cfg.task.update({"mode": "non_whiten", "prompt": "2 3", "neuronpedia": {"enabled": False}})
    cfg.task.capture.selection = {"candidate_count": "all"}
    cfg.task.capture.output = {"feature_count": 2}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(operation, "_pair", lambda cfg: pair)
    monkeypatch.setattr(operation, "_prompt", lambda cfg: cfg.task.prompt)
    single = operation.execute(cfg)
    assert single["model_answer"]["status"] == "completed"
    assert len(single["top_features"]) == 2
    assert single["completion"]["status"] == "complete"
    cfg.task.output_scope = "all_tokens"
    cfg.task.max_tokens = 32
    cfg.task.max_active_per_token = 8
    cfg.task.capture.output.feature_count = None
    all_tokens = operation.execute(cfg)
    assert len(all_tokens["tokens"]) == 3
    assert len(all_tokens["top_features"]) == 4
    assert all_tokens["model_answer"]["text"] == single["model_answer"]["text"]
    cfg.task.output_scope = "last_token"
    cfg.task.capture.output.feature_count = 2
    cfg.task.prompt = ""
    monkeypatch.setattr(
        datasets,
        "select_dataset",
        lambda *a, **kw: datasets.SampleSelection(
            [{"sample_id": "q1", "prompt": "2 3"}, {"sample_id": "q2", "prompt": "4 5"}],
            {"selected_count": 2},
        ),
    )
    batch = operation.execute(cfg)
    assert batch["stages"]["model_answer"]["completed"] == 2
    assert batch["completion"]["completed"] == 2
    ledger = [json.loads(line) for line in (tmp_path / "reports/capture_samples.jsonl").open()]
    assert ledger[0]["result"]["model_answer"]["reused"]
    assert ledger[0]["result"]["top_features"] == single["top_features"]
    # Native dataset report consumes the answer from the normal result reader.
    from tasks.capture.report import export_report

    (tmp_path / "metadata/manifest.json").write_text(
        json.dumps(
            {
                "schema": "saebase.experiment",
                "task": "capture",
                "status": "executed",
                "configuration": OmegaConf.to_container(cfg),
                "result": EvaluationResult.from_payload("capture", cfg, batch).to_dict(),
            }
        )
    )
    monkeypatch.setattr(
        datasets,
        "read_selection",
        lambda *a: [{"sample_id": "q1", "prompt": "2 3"}, {"sample_id": "q2", "prompt": "4 5"}],
    )
    export_report(tmp_path)
    report = json.loads((tmp_path / "reports/question_features_pairs.json").read_text())
    assert report["questions"][0]["model_answer"]["text"] == single["model_answer"]["text"]
    assert "Model Original Answer" in (tmp_path / "reports/question_features_pairs.md").read_text()
    assert "Model Original Answer" in (tmp_path / "reports/question_features_pairs.html").read_text()

    from tasks.capture.task import CaptureTask
    import saebase.runtime.sae_operations as runtime
    from tasks.capture.annotations import refresh_annotations

    ledger_path = tmp_path / "reports/capture_samples.jsonl"
    original_ledger = ledger_path.read_bytes()
    monkeypatch.setattr(runtime, "_pair", lambda cfg: pair)
    monkeypatch.setattr(
        operation,
        "_capture_features",
        lambda *a, **kw: pytest.fail("Resume must retain feature measurements"),
    )
    cfg.runtime = {"mode": "resume"}
    resumed = CaptureTask().run_validated(cfg)
    assert resumed.completion["status"] == "complete"
    assert ledger_path.read_bytes() == original_ledger
    OmegaConf.save(cfg, tmp_path / "metadata/config.yaml")
    generate_answers = owner.answers
    monkeypatch.setattr(
        owner,
        "answers",
        lambda cfg, pair, prompts: [
            {"status": "failed", "error": {"message": "generation unavailable"}} for _ in prompts
        ],
    )
    owner.refresh(tmp_path, pair=pair)
    refreshed = refresh_annotations(tmp_path)
    assert refreshed["annotation"]["status"] == "complete"
    assert refreshed["completion"]["status"] == "incomplete"
    manifest = json.loads((tmp_path / "metadata/manifest.json").read_text())
    assert (
        CaptureTask().read_result(tmp_path, manifest)["result"]["completion"]["status"]
        == "incomplete"
    )
    monkeypatch.setattr(owner, "answers", generate_answers)
    owner.refresh(tmp_path, pair=pair)
    assert (
        CaptureTask().read_result(tmp_path, manifest)["result"]["completion"]["status"]
        == "complete"
    )
    assert ledger_path.read_bytes() == original_ledger


def test_one_invalid_input_does_not_block_other_answers(cfg, pair):
    result = owner.answers(cfg, pair, ["2 3", "not a token", "4 5"])
    assert [r["status"] for r in result] == ["completed", "failed", "completed"]


def test_rss_variant_answer_failure_keeps_measurement_but_blocks_completion():
    base = {
        "completion": completion_summary([{"status": "completed"}]),
        "rss_features": [{"feature_id": 7}],
    }
    result = owner.attach(
        base,
        {"status": "completed", "text": "clean"},
        extra_answers=[{"status": "failed", "error": {"message": "variant generation failed"}}],
    )
    assert result["rss_features"] == [{"feature_id": 7}]
    assert result["stages"]["model_answer"]["requested"] == 2
    assert result["completion"]["status"] == "incomplete"


def test_rss_answer_only_refresh_covers_every_variant(cfg, pair, tmp_path, monkeypatch):
    from saebase.store import sha256_state_dict
    from saebase.results import EvaluationResult
    from tasks.capture.task import CaptureTask
    import saebase.runtime.sae_operations as runtime

    cfg.model = {"name": "tiny"}
    cfg.dataset = {"name": "synthetic"}
    cfg.task.prompt = "2 3"
    original = {
        "status": "executed",
        "operation": "capture",
        "mode": "rss",
        "clean_prompt": "2 3",
        "variants": ["4 5"],
        "input_identity": {"model_sha256": sha256_state_dict(pair.model.model)},
        "prompt_sha256": hashlib.sha256(b"2 3").hexdigest(),
        "completion": completion_summary([{"status": "completed"}]),
        "rss_features": [{"feature_id": 7}],
    }
    manifest = {
        "task": "capture",
        "status": "executed",
        "configuration": OmegaConf.to_container(cfg),
        "result": EvaluationResult.from_payload("capture", cfg, original).to_dict(),
    }
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, "_pair", lambda cfg: pair)
    update = CaptureTask().refresh_model_answers(tmp_path)
    assert update["model_answer"]["completed"] == 2
    result = CaptureTask().read_result(tmp_path, manifest)["result"]["summary"]
    assert result["model_answer"]["status"] == "completed"
    assert result["variant_model_answers"][0]["status"] == "completed"
    assert result["rss_features"] == [{"feature_id": 7}]


def test_answer_for_different_tokens_cannot_be_attached_as_completed():
    result = {
        "input_identity": {"data_sha256": "wrong input"},
        "top_features": [{"feature_id": 7}],
        "completion": completion_summary([{"status": "completed"}]),
    }
    answer = {
        "status": "completed",
        "text": "another input answer",
        "identity": {"prompt_token_ids": [1, 2], "prompt_sha256": "other"},
    }
    owner.attach(result, answer)
    assert result["model_answer"]["status"] == "blocked"
    assert result["completion"]["status"] == "incomplete"
    assert result["top_features"] == [{"feature_id": 7}]
