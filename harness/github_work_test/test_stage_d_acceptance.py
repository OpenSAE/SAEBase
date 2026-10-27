"""CPU unit tests for Stage D scheduling, resume and acceptance contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from saebase.sae.coverage import load_coverage
from saebase.sae.stage_d import (
    PRIOR_PAIR_VERIFIED,
    SCHEMA_ACCEPTANCE,
    build_final_matrix,
    build_site_records,
    canonical_payload_sha256,
    classify_exception,
    load_stage_d_targets,
    read_acceptance,
    schedule_sites,
    should_skip_resume,
    update_status,
    verify_pair_site,
    write_acceptance,
    write_stage_d_targets,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "conf" / "coverage" / "stage_bc.json"


class TinyModel(torch.nn.Module):
    """TransformerLens-like toy used by Stage D protocol tests."""

    def __init__(self, hook_name: str = "blocks.12.hook_resid_post"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(1, 13, dtype=torch.float).reshape(3, 4) / 10)
        self.cfg = SimpleNamespace(n_layers=24, device="cpu", d_model=3)
        self.hook_name = hook_name

    def to_tokens(self, text, prepend_bos=True):
        vals = [((ord(c) % 9) + 1) for c in text[:6]] or [1, 2, 3]
        return torch.tensor([vals], dtype=torch.long)

    def _hidden(self, tokens):
        x = tokens.float().unsqueeze(-1)
        return torch.cat([x / 7, x / 11 + 1, x / 13 + 2], dim=-1)

    def _out(self, h):
        return torch.cumsum(h, dim=1) @ self.weight

    def forward(self, tokens):
        return self._out(self._hidden(tokens))

    def run_with_cache(self, tokens, names_filter=None):
        return self(tokens), {names_filter: self._hidden(tokens)}

    def run_with_hooks(self, tokens, fwd_hooks):
        h = self._hidden(tokens)
        for name, fn in fwd_hooks:
            if name != self.hook_name:
                raise ValueError(f"unknown hook {name}")
            h = fn(h, None)
        return self._out(h)


class AffineSAE(torch.nn.Module):
    def __init__(self, *, d_in=3, d_sae=5):
        super().__init__()
        self.cfg = SimpleNamespace(d_in=d_in, d_sae=d_sae, dtype="float32")
        self.decoder_weight = torch.nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.1, 0.2, -0.1],
                    [0.1, -0.1, 0.3],
                ]
            )
        )
        self.bias = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.1]))

    def encode(self, x):
        return torch.cat(
            [x, torch.ones((*x.shape[:-1], 2), dtype=x.dtype, device=x.device)],
            -1,
        )

    def decode(self, z):
        return z @ self.decoder_weight + self.bias


def test_targets_cover_exactly_48_unique_sites(tmp_path):
    path = tmp_path / "stage_d_targets.json"
    payload = write_stage_d_targets(path, load_coverage(CATALOG))
    records = load_stage_d_targets(path)
    assert payload["n_sites"] == 48
    assert len(records) == 48
    assert len({r.key.id for r in records}) == 48
    assert sum(r.prior_pair_verified for r in records) == 4
    assert len(PRIOR_PAIR_VERIFIED) == 4


def test_schedule_skips_prior_unless_explicit():
    records = build_site_records(load_coverage(CATALOG))
    queued = schedule_sites(records)
    assert len(queued) == 44
    assert all(not r.prior_pair_verified for r in queued)
    d1 = schedule_sites(records, sae_key="qwen35_2b_resid_w32k", layer=12)
    assert len(d1) == 1 and d1[0].prior_pair_verified
    family = schedule_sites(records, model="qwen35_2b")
    assert [r.layer for r in family] == [6, 15, 20]
    all_prior = schedule_sites(records, include_prior=True)
    assert len(all_prior) == 48
    assert len({r.key.id for r in all_prior}) == 48


def test_schedule_filters_and_model_order():
    records = build_site_records(load_coverage(CATALOG))
    one = schedule_sites(records, model="pythia410m", layer=12)
    assert len(one) == 1 and one[0].layer == 12
    ordered = schedule_sites(records, include_prior=True)
    models = []
    for rec in ordered:
        if not models or models[-1] != rec.model_key:
            models.append(rec.model_key)
    assert models.index("qwen35_2b") < models.index("qwen3_8b")
    assert models.index("pythia410m") < models.index("qwen35_9b")


def test_acceptance_checksum_and_resume(tmp_path):
    payload = {
        "schema": SCHEMA_ACCEPTANCE,
        "status": "passed",
        "model_key": "qwen35_2b",
        "sae_key": "qwen35_2b_resid_w32k",
        "layer": 12,
        "hook_name": "blocks.12.hook_resid_post",
        "model_state_sha256": "a" * 64,
        "sae_state_sha256": "b" * 64,
        "d_in": 4,
        "d_sae": 8,
        "n_prompts": 1,
        "identity_patch_max_abs_logit_error": 0.0,
        "feature_ablation_logit_l2": [1.0],
        "peak_cuda_allocated_bytes": 0,
    }
    payload["artifact_sha256"] = canonical_payload_sha256(payload)
    path = write_acceptance(tmp_path, payload)
    assert read_acceptance(path)["artifact_sha256"] == payload["artifact_sha256"]
    assert (
        should_skip_resume(
            tmp_path,
            model_key="qwen35_2b",
            sae_key="qwen35_2b_resid_w32k",
            layer=12,
            resume=True,
        )
        is not None
    )
    assert (
        should_skip_resume(
            tmp_path,
            model_key="qwen35_2b",
            sae_key="qwen35_2b_resid_w32k",
            layer=12,
            resume=False,
        )
        is None
    )
    bad = dict(payload)
    bad["artifact_sha256"] = "c" * 64
    bad_path = path.with_name("L12_bad.json")
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_acceptance(bad_path)


def test_empty_and_invalid_outputs_rejected(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_acceptance(empty)


def test_classify_exception_vocabulary():
    assert classify_exception(RuntimeError("CUDA out of memory")) == "blocked_compute"
    assert classify_exception(PermissionError("401 gated")) == "blocked_access"
    assert classify_exception(ImportError("no module")) == "blocked_loader"
    assert classify_exception(ValueError("identity patch maximum logit error")) == "failed_numeric"
    assert classify_exception(ValueError("manifest integrity failed")) == "failed_integrity"


def test_verify_pair_site_with_toy_loader():
    hook = "blocks.12.hook_resid_post"
    model = TinyModel(hook)
    sae = AffineSAE()
    meta = {"hook_name": hook, "loaded_model_name": "tiny", "sae_id": "toy"}

    def load_fn(sae_key, layer, device=None, model_key_override=None):
        assert sae_key == "qwen35_2b_resid_w32k" and layer == 12
        return model, sae, meta

    record = verify_pair_site(
        sae_key="qwen35_2b_resid_w32k",
        layer=12,
        device="cpu",
        load_fn=load_fn,
    )
    assert record["status"] == "passed"
    assert record["schema"] == SCHEMA_ACCEPTANCE
    assert record["n_prompts"] == 3
    assert record["identity_patch_max_abs_logit_error"] <= 1e-2
    assert all(x > 1e-8 for x in record["feature_ablation_logit_l2"])


def test_verify_detects_hook_mismatch():
    model = TinyModel()
    sae = AffineSAE()

    def load_fn(sae_key, layer, device=None, model_key_override=None):
        return model, sae, {"hook_name": "blocks.12.hook_resid_pre"}

    with pytest.raises(ValueError, match="hook"):
        verify_pair_site(
            sae_key="qwen35_2b_resid_w32k",
            layer=12,
            device="cpu",
            load_fn=load_fn,
        )


def test_verify_detects_dimension_mismatch():
    model = TinyModel()
    sae = AffineSAE(d_in=3, d_sae=5)
    sae.cfg = SimpleNamespace(d_in=99, d_sae=5, dtype="float32")

    def load_fn(sae_key, layer, device=None, model_key_override=None):
        return (
            model,
            sae,
            {
                "hook_name": "blocks.12.hook_resid_post",
                "loaded_model_name": "tiny",
            },
        )

    with pytest.raises(ValueError, match="shape|dimension"):
        verify_pair_site(
            sae_key="qwen35_2b_resid_w32k",
            layer=12,
            device="cpu",
            load_fn=load_fn,
        )


def test_verify_detects_noop_feature_edit(monkeypatch):
    hook = "blocks.12.hook_resid_post"
    model = TinyModel(hook)
    sae = AffineSAE()
    meta = {"hook_name": hook, "loaded_model_name": "tiny", "sae_id": "toy"}

    def load_fn(sae_key, layer, device=None, model_key_override=None):
        return model, sae, meta

    import saebase.api as api

    def fake_edit(sae_obj, snap, replacements):
        return snap

    monkeypatch.setattr(api, "edit_features", fake_edit)
    with pytest.raises(ValueError, match="ablation"):
        verify_pair_site(
            sae_key="qwen35_2b_resid_w32k",
            layer=12,
            device="cpu",
            load_fn=load_fn,
        )


def test_status_and_final_matrix(tmp_path):
    records = build_site_records(load_coverage(CATALOG))
    site = records[0]
    update_status(
        tmp_path,
        site.key.id,
        {
            "status": "blocked_access",
            "pair_verified": False,
            "error_type": "GatedRepoError",
            "error": "401",
        },
    )
    matrix = build_final_matrix(tmp_path, records)
    assert matrix["n_sites"] == 48
    assert matrix["counts"].get("blocked_access") == 1


def test_checked_in_targets_match_catalog():
    path = ROOT / "conf" / "coverage" / "stage_d_targets.json"
    assert path.exists()
    records = load_stage_d_targets(path)
    live = build_site_records(load_coverage(CATALOG))
    assert {r.key.id for r in records} == {r.key.id for r in live}


def test_matrix_dry_run_lists_44(tmp_path):
    from saebase.tools.phase_d_matrix import main

    code = main(
        [
            "--dry-run",
            "--output",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    body = json.loads((tmp_path / "stage_d_targets.json").read_text())
    assert body["n_sites"] == 48


def test_resolve_evaluation_host_auto_override_for_transfer():
    from saebase.sae.stage_d import resolve_evaluation_host

    host, override = resolve_evaluation_host(
        catalog_model_key="gemma2_2b_it",
        sae_key="gemma2_2b_it_matryoshka_l12_32k",
    )
    assert host == "gemma2_2b_it"
    assert override == "gemma2_2b_it"


def test_assert_host_rejects_base_labeled_as_it():
    from saebase.sae.stage_d import assert_host_matches_label

    with pytest.raises(ValueError, match="IT|instruct|base"):
        assert_host_matches_label(
            evaluation_host_model_key="gemma2_2b_it",
            loaded_model_name="google/gemma-2-2b",
        )


def test_population_corpus_has_64_groups_and_no_overlap():
    from saebase.sae.stage_d import (
        DEFAULT_POPULATION_PATH,
        POPULATION_MIN_EVAL_GROUPS,
        load_population_corpus,
    )

    corpus = load_population_corpus(DEFAULT_POPULATION_PATH)
    assert corpus["n_evaluation_groups"] >= POPULATION_MIN_EVAL_GROUPS
    assert corpus["n_selection"] >= 16
    sel = {r["prompt"] for r in corpus["selection"]}
    ev = {r["prompt"] for r in corpus["evaluation"]}
    assert not sel & ev


def test_population_flag_rejects_tiny_corpus(tmp_path):
    from saebase.sae.stage_d import load_population_corpus

    tiny = tmp_path / "tiny.jsonl"
    rows = [
        {"split": "selection", "group_id": "s0", "prompt": "sel A"},
        {"split": "evaluation", "group_id": "e0", "prompt": "eval A"},
        {"split": "evaluation", "group_id": "e1", "prompt": "eval B"},
    ]
    tiny.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation groups"):
        load_population_corpus(tiny)


def test_resume_fingerprint_mismatch_forces_rerun(tmp_path):
    payload = {
        "schema": SCHEMA_ACCEPTANCE,
        "status": "passed",
        "model_key": "qwen35_2b",
        "evaluation_host_model_key": "qwen35_2b",
        "sae_key": "qwen35_2b_resid_w32k",
        "layer": 12,
        "hook_name": "blocks.12.hook_resid_post",
        "protocol": "stage_d.pair.v2",
        "model_state_sha256": "a" * 64,
        "sae_state_sha256": "b" * 64,
        "d_in": 4,
        "d_sae": 8,
        "n_prompts": 1,
        "identity_patch_max_abs_logit_error": 0.0,
        "feature_ablation_logit_l2": [1.0],
        "peak_cuda_allocated_bytes": 0,
    }
    payload["artifact_sha256"] = canonical_payload_sha256(payload)
    write_acceptance(tmp_path, payload)
    assert (
        should_skip_resume(
            tmp_path,
            model_key="qwen35_2b",
            sae_key="qwen35_2b_resid_w32k",
            layer=12,
            resume=True,
            expected={"hook_name": "blocks.12.hook_resid_post", "protocol": "stage_d.pair.v2"},
        )
        is not None
    )
    assert (
        should_skip_resume(
            tmp_path,
            model_key="qwen35_2b",
            sae_key="qwen35_2b_resid_w32k",
            layer=12,
            resume=True,
            expected={"hook_name": "blocks.12.hook_resid_pre", "protocol": "stage_d.pair.v2"},
        )
        is None
    )


def test_layer_norm_snapshot_restores_and_clears():
    """Synthetic SAELens-like layer_norm encode/decode with per-input stats."""
    from saebase.api import ActivationSite, ModelAdapter, capture_features, edit_features

    class LayerNormTopK(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(
                d_in=4, d_sae=4, dtype="float32", normalize_activations="layer_norm"
            )
            self.W_enc = torch.nn.Parameter(torch.eye(4))
            self.W_dec = torch.nn.Parameter(torch.eye(4))
            self.b_dec = torch.nn.Parameter(torch.zeros(4))
            self.k = 2

        def process_sae_in(self, x):
            mu = x.mean(dim=-1, keepdim=True)
            xc = x - mu
            std = xc.std(dim=-1, keepdim=True).clamp_min(1e-5)
            self.ln_mu = mu
            self.ln_std = std
            return xc / std

        def encode(self, x):
            x = self.process_sae_in(x)
            z = x @ self.W_enc
            # TopK
            vals, idx = torch.topk(z.abs(), k=self.k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, idx, 1.0)
            return z * mask

        def decode(self, z):
            out = z @ self.W_dec + self.b_dec
            out = out * self.ln_std + self.ln_mu
            del self.ln_mu
            del self.ln_std
            return out

    class LNModel(TinyModel):
        def __init__(self):
            super().__init__("blocks.0.hook_mlp_out")
            self.cfg = SimpleNamespace(n_layers=1, device="cpu", d_model=4)
            self.weight = torch.nn.Parameter(torch.randn(4, 4))

        def _hidden(self, tokens):
            # Vary activation scale by prompt content for distinct ln stats.
            base = tokens.float().unsqueeze(-1)
            scale = 1.0 + (tokens.float().sum() % 7) / 10.0
            return torch.cat([base, base + 1, base + 2, base * scale], dim=-1)[..., :4]

        def _out(self, h):
            return h @ self.weight

        def run_with_cache(self, tokens, names_filter=None):
            return self(tokens), {names_filter: self._hidden(tokens)}

        def run_with_hooks(self, tokens, fwd_hooks):
            h = self._hidden(tokens)
            for name, fn in fwd_hooks:
                if name != self.hook_name:
                    raise ValueError(name)
                h = fn(h, None)
            return self._out(h)

    model = ModelAdapter(LNModel())
    sae = LayerNormTopK()
    site = ActivationSite(0, "blocks.0.hook_mlp_out", position=-1)
    prompts = ["alpha prompt one", "beta prompt two distinctly longer"]
    snaps = []
    for prompt in prompts:
        snap = capture_features(model, sae, site=site, prompt=prompt)
        assert snap.ln_mu is not None and snap.ln_std is not None
        assert not hasattr(sae, "ln_mu") and not hasattr(sae, "ln_std")
        # Reference encode path (manual) must match TopK feature indices.
        h = snap.activation.to(dtype=torch.float32).unsqueeze(0)
        z_ref = sae.encode(h)[0]
        assert torch.equal(
            torch.topk(snap.codes.abs(), k=2).indices.sort().values,
            torch.topk(z_ref.abs(), k=2).indices.sort().values,
        )
        # Encode leaves ln_* buffers; clear them like an unused decode would.
        for name in ("ln_mu", "ln_std"):
            if hasattr(sae, name):
                delattr(sae, name)
        _ = edit_features(sae, snap, replacements={int(snap.codes.abs().argmax()): 0.0})
        assert not hasattr(sae, "ln_mu") and not hasattr(sae, "ln_std")
        snaps.append(snap)
    # Distinct prompts must not share LN stats.
    assert not torch.allclose(snaps[0].ln_mu, snaps[1].ln_mu) or not torch.allclose(
        snaps[0].ln_std, snaps[1].ln_std
    )


def test_verify_evidence_fails_on_tampered_checksum(tmp_path):
    from saebase.tools.phase_d_verify_evidence import verify_checksums

    final = tmp_path / "final"
    final.mkdir()
    f = tmp_path / "notes.txt"
    f.write_text("ok\n", encoding="utf-8")
    digest = __import__("hashlib").sha256(b"ok\n").hexdigest()
    manifest = final / "checksums.sha256"
    manifest.write_text(f"{digest}  notes.txt\n", encoding="utf-8")
    assert verify_checksums(manifest, tmp_path) == []
    f.write_text("tampered\n", encoding="utf-8")
    errs = verify_checksums(manifest, tmp_path)
    assert errs and "mismatch" in errs[0]


def test_rebuild_canonical_status_is_exactly_48(tmp_path):
    from saebase.sae.stage_d import rebuild_canonical_status

    records = build_site_records(load_coverage(CATALOG))
    update_status(tmp_path, "bogus|extra|L0|blocks.0.hook_resid_pre", {"status": "failed_numeric"})
    update_status(tmp_path, records[0].key.id, {"status": "pair_verified", "pair_verified": True})
    payload = rebuild_canonical_status(tmp_path, records)
    assert payload["n_sites"] == 48
    assert len(payload["sites"]) == 48
    assert "bogus|extra|L0|blocks.0.hook_resid_pre" not in payload["sites"]
    assert (tmp_path / "status_archive_extras.json").exists()
