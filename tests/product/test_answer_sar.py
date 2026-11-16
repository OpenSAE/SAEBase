"""Answer-aligned SAR contracts using causal synthetic logits, not benchmark results."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from saebase.datasets.answers import ReferenceAnswerUnavailable, reference_answer

from saebase.api import ActivationSite, ModelAdapter, SAEIdentity, capture_features, edit_features
from tests.fixtures.synthetic import ResearchToyModel, ResearchToySAE


def test_frozen_prefix_matches_full_scoring_and_stays_immutable():
    tl = pytest.importorskip("transformer_lens")
    from saebase.api.answer_cache import FrozenPrefixAnswerReadout

    torch.manual_seed(7)
    raw = tl.HookedTransformer(
        tl.HookedTransformerConfig(
            n_layers=2,
            d_model=8,
            d_head=4,
            n_heads=2,
            d_mlp=16,
            n_ctx=32,
            d_vocab=17,
            act_fn="relu",
            device="cpu",
        )
    ).eval()
    adapter = ModelAdapter(raw)
    prompt, answer = torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[6, 7, 8]])
    site = ActivationSite(0, "blocks.0.hook_resid_post")
    activation = adapter.capture(tokens=prompt, site=site)
    scorer = FrozenPrefixAnswerReadout(adapter, prompt, answer, site)
    original = [entry.past_keys.clone() for entry in scorer.cache.entries]
    validation = scorer.verify([activation, activation + 0.5], atol=1e-6, rtol=1e-6)
    assert validation["max_abs_error"] < 1e-6
    scorer(activation - 0.2)
    for entry, expected in zip(scorer.cache.entries, original):
        torch.testing.assert_close(entry.past_keys, expected, rtol=0, atol=0)


def test_teacher_forcing_scores_all_correct_shifted_tokens_and_patches_prompt_only():
    raw = ResearchToyModel().eval()
    model = ModelAdapter(raw)
    prompt, answer = torch.tensor([[1, 2, 3]]), torch.tensor([[0, 1]])
    joint = torch.cat((prompt, answer[:, :-1]), 1)
    expected = raw(joint)[0, 2:].float().log_softmax(-1).gather(1, answer[0, :, None]).squeeze(1)
    actual = model.answer_token_logprobs(prompt_tokens=prompt, answer_tokens=answer)
    torch.testing.assert_close(actual, expected)
    site = ActivationSite(0, raw.site_name)
    activation = model.capture(tokens=prompt, site=site)
    replay = model.answer_token_logprobs(
        prompt_tokens=prompt, answer_tokens=answer, site=site, activation=activation
    )
    torch.testing.assert_close(replay, expected)
    edited = activation + 0.3
    hidden = raw._hidden(joint)
    hidden[0, 2] = edited
    manually_patched = (
        raw._logits(hidden)[0, 2:].float().log_softmax(-1).gather(1, answer[0, :, None]).squeeze(1)
    )
    actual_patch = model.answer_token_logprobs(
        prompt_tokens=prompt, answer_tokens=answer, site=site, activation=edited
    )
    torch.testing.assert_close(actual_patch, manually_patched)
    assert actual_patch.shape == (2,) and not torch.equal(actual_patch, replay)


@pytest.mark.parametrize(
    "dataset,row,expected",
    [
        ("HotpotQA", {"source_record": {"answer": "John Doman"}}, "John Doman"),
        ("GSM8K", {"source_record": {"answer": "reasoning\n#### 42"}}, "42"),
        (
            "BBH",
            {"source_record": {"target": "(B)", "input": "Question\n(A) red\n(B) blue"}},
            "blue",
        ),
        ("IOI", {"target": {"correct_text": "Alice"}}, "Alice"),
        ("RAVEL", {"reference_answer": "Paris"}, "Paris"),
        (
            "CounterFact",
            {
                "source_record": {
                    "requested_rewrite": {
                        "target_true": {"str": "French"},
                        "target_new": {"str": "Russian"},
                    }
                }
            },
            "French",
        ),
        (
            "ARCChallenge",
            {
                "source_record": {
                    "answerKey": "B",
                    "choices": {"label": ["A", "B"], "text": ["red", "blue"]},
                }
            },
            "blue",
        ),
        (
            "MMLUPro",
            {"source_record": {"answer": "B", "answer_index": 1, "options": ["red", "blue"]}},
            "blue",
        ),
        (
            "TruthfulQA",
            {"source_record": {"mc1_targets": {"choices": ["false", "true"], "labels": [0, 1]}}},
            "true",
        ),
        (
            "HumanEval",
            {
                "prompt": 'def f():\n    """doc"""',
                "source_record": {"canonical_solution": "    return 1\n"},
            },
            "    return 1\n",
        ),
    ],
)
def test_reference_answers_are_derived_from_actual_label_fields(dataset, row, expected):
    before = json.dumps(row)
    result = reference_answer(row, dataset)
    assert result["text"] == expected and json.dumps(row) == before
    assert result["source_field"] and result["prefix"] == ("\n" if dataset == "HumanEval" else " ")


@pytest.mark.parametrize("dataset", ["HarmBench", "XSTest"])
def test_safety_classification_labels_are_not_fabricated_reference_responses(dataset):
    with pytest.raises(ReferenceAnswerUnavailable):
        reference_answer({"source_record": {"label": "safe", "Behavior": "a prompt"}}, dataset)
    assert reference_answer({"reference_answer": "An explicitly supplied reference"}, dataset)[
        "text"
    ]


@pytest.mark.parametrize("context_shift", [0.0, 0.02])
def test_answer_sar_reuses_prompt_capture_and_original_estimator(
    tmp_path, monkeypatch, context_shift
):
    from tasks.capture import operation
    from tests.product.test_capture_sar import PAPER_DOUBLE

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "sar.py"
    source.write_text(PAPER_DOUBLE)
    raw = ResearchToyModel().eval()
    hidden = raw._hidden
    raw._hidden = lambda ids: hidden(ids) + (context_shift if ids.shape[1] > 3 else 0.0)
    raw.tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: [int(x) for x in text.split()]
    )
    pair = SimpleNamespace(
        model=ModelAdapter(raw),
        sae=ResearchToySAE().eval(),
        identity=SAEIdentity(
            "synthetic", "synthetic", "synthetic", "synthetic", raw.site_name, 0, "synthetic", 4, 7
        ),
    )
    pair.site = pair.identity.site
    monkeypatch.setattr(operation, "_pair", lambda cfg: pair)
    import tasks.sae_runtime as runtime

    monkeypatch.setattr(runtime, "pair_state_fingerprints", lambda _: ("a" * 64, "b" * 64))
    cfg = OmegaConf.create(
        {
            "seed": 3,
            "dataset": {"name": "explicit"},
            "task": {
                "name": "capture",
                "prompt": "1 2 3",
                "mode": "sar",
                "top_feature_num": 3,
                "neuronpedia": {"enabled": False},
                "sar": {
                    "source": str(source),
                    "readout": "answer_mean_logprob",
                    "answer": "0 1",
                    "M": 32,
                    "n_fit": 24,
                },
            },
        }
    )
    captured = []
    original = pair.model.capture

    def capture(*, tokens, site):
        captured.append(tokens.clone())
        return original(tokens=tokens, site=site)

    monkeypatch.setattr(pair.model, "capture", capture)
    result = operation.execute(cfg)
    assert len(captured) == 2 and captured[0].tolist() == [[1, 2, 3]]
    assert captured[1].tolist() == [[1, 2, 3, 0]]
    sar = result["sar"]
    assert sar["identity"]["behavior"]["answer_token_ids"] == [0, 1]
    assert sar["identity"]["behavior"]["feature_selection_uses_answer"] is False
    with np.load(sar["artifact"]["path"]) as data:
        np.testing.assert_array_equal(data["source_tokens"], [[1, 2, 3]])
        np.testing.assert_array_equal(data["answer_token_ids"], [0, 1])
        snapshot = capture_features(pair.model, pair.sae, prompt="1 2 3", site=pair.site)
        for mask, measured in zip(data["masks"], data["responses"]):
            patch = edit_features(
                pair.sae,
                snapshot,
                replacements={i: 0.0 for i, keep in zip(sar["feature_ids"], mask) if not keep},
            )
            expected = (
                pair.model.answer_token_logprobs(
                    prompt_tokens=snapshot.tokens,
                    answer_tokens=torch.tensor([[0, 1]]),
                    site=pair.site,
                    activation=raw._hidden(torch.tensor([[1, 2, 3, 0]]))[0, 2]
                    + patch.activation
                    - snapshot.activation,
                )
                .double()
                .mean()
            )
            assert measured == pytest.approx(float(expected))
    assert result["completion"]["status"] == "complete"


def test_hf_backend_full_teacher_forced_output_preserves_single_position_api():
    from tests.product.test_capture_with_logits import _hooks

    model, _ = _hooks()
    ids = torch.tensor([[1, 2, 3]])
    h = model.get_resid(ids, 0, [0])[0]
    old = model.run_with_patch(ids, 0, {0: h})
    full = model.run_with_patch(ids, 0, {0: h}, readout_offset=None)
    assert full.shape == (1, 3, 5)
    torch.testing.assert_close(full[0, -1], old)
    adapter = ModelAdapter(model, backend="hf_hooks")
    values = adapter.answer_token_logprobs(
        prompt_tokens=ids,
        answer_tokens=torch.tensor([[0, 1]]),
        site=ActivationSite(0, "blocks.0.hook_resid_post"),
        activation=h,
    )
    assert values.shape == (2,) and torch.isfinite(values).all()


def test_full_answer_context_is_not_silently_truncated():
    from saebase.api import UnsupportedCapability

    raw = ResearchToyModel()
    raw.cfg.n_ctx = 3
    model = ModelAdapter(raw)
    with pytest.raises(UnsupportedCapability, match="context window"):
        model.answer_token_logprobs(
            prompt_tokens=torch.tensor([[1, 2, 3]]), answer_tokens=torch.tensor([[0, 1]])
        )
