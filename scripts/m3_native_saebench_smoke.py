#!/usr/bin/env python3
"""Run *native upstream* SAEBench Core and Sparse Probing, bounded to one CPU SAE.

No metric estimator is copied into SAEBase. The upstream repo must be an
editable installation and have commit 8042bb3828c6340da8d12062324e92b2077c571c.
The two native protocols use the same Pythia model and same SAE checkpoint,
with their own documented dataset and metric configurations.
"""
from __future__ import annotations
import json
from pathlib import Path
import random
import os

import numpy as np
import torch

from sae_bench.evals.core import main as upstream_core
from sae_bench.evals.sparse_probing import main as upstream_sparse
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.sae_bench_utils.general_utils import get_results_filepath

from saebase.benchmarks.adapters.saebench import import_native

RELEASE = "sae_bench_pythia70m_sweep_gated_ctx128_0730"
SAE_ID = "blocks.3.hook_resid_post__trainer_5"
MODEL = "pythia-70m-deduped"
DATASET = "fancyzhx/ag_news"
COMMIT = "8042bb3828c6340da8d12062324e92b2077c571c"

def validate(path: Path, metric: str, native: dict) -> float:
    value, digest, _ = import_native(
        path, metric, expected_sae_id=SAE_ID,
        expected_model_name=MODEL, expected_sae_release=RELEASE,
        expected_commit=COMMIT,
    )
    assert len(digest) == 64
    path_keys = {"saebench.core.l0": ("sparsity","l0"),
                 "saebench.core.mse": ("reconstruction_quality","mse"),
                 "saebench.sparse_probing.sae_top_1_test_accuracy":
                     ("sae","sae_top_1_test_accuracy")}[metric]
    expected = native["eval_result_metrics"]
    for key in path_keys:
        expected = expected[key]
    assert value == float(expected), (metric, value, expected)
    return value

def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(44)
    random.seed(44)
    np.random.seed(44)
    output = Path(os.environ.get("SAEBASE_NATIVE_RESULTS", "native-results")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Upstream get_sae_bench_version() shells out to git in the current
    # directory. Without this, provenance would mistakenly identify SAEBase.
    upstream = (Path.cwd() / "upstream").resolve()
    if not (upstream / ".git").exists():
        raise FileNotFoundError("expected a pinned SAEBench checkout at ./upstream")
    os.chdir(upstream)
    core_root = output / "core"
    sparse_root = output / "sparse_probing"

    # Official upstream API. One batch for core variance/reconstruction,
    # unchanged upstream estimator and published SAE. Explicitly bounded.
    core = upstream_core.multiple_evals(
        selected_saes=[(RELEASE, SAE_ID)],
        n_eval_reconstruction_batches=1,
        n_eval_sparsity_variance_batches=1,
        eval_batch_size_prompts=2,
        compute_featurewise_density_statistics=False,
        compute_featurewise_weight_based_metrics=False,
        exclude_special_tokens_from_reconstruction=False,
        dataset=DATASET, context_size=32, output_folder=str(core_root),
        device="cpu", dtype="float32", force_rerun=True,
    )
    assert len(core) == 1, f"expected one native core result; got {len(core)}"
    core_path = Path(get_results_filepath(str(core_root), RELEASE, SAE_ID))
    core_native = json.loads(core_path.read_text())
    assert core_native["eval_config"]["dataset"] == DATASET
    assert core_native["eval_config"]["context_size"] == 32
    core_metrics = {
        metric: validate(core_path, metric, core_native)
        for metric in ("saebench.core.l0", "saebench.core.mse")
    }

    # Official upstream API and original train/test/probe implementation.
    # The small sample budget keeps the CPU regression test bounded.
    cfg = SparseProbingEvalConfig(
        model_name=MODEL, dataset_names=[DATASET],
        random_seed=44, probe_train_set_size=64,
        probe_test_set_size=32, context_length=32,
        sae_batch_size=64, llm_batch_size=4, llm_dtype="float32",
        k_values=[1], lower_vram_usage=True,
    )
    sparse = upstream_sparse.run_eval(
        cfg, [(RELEASE, SAE_ID)], "cpu", str(sparse_root),
        force_rerun=True, clean_up_activations=True,
        save_activations=False, artifacts_path=str(output / "temporary"),
    )
    assert len(sparse) == 1, f"expected one native Sparse Probing result: {list(sparse)}"
    sparse_path = Path(get_results_filepath(str(sparse_root), RELEASE, SAE_ID))
    sparse_native = json.loads(sparse_path.read_text())
    assert sparse_native["eval_config"]["dataset_names"] == [DATASET]
    assert sparse_native["eval_config"]["random_seed"] == 44
    sparse_metric = "saebench.sparse_probing.sae_top_1_test_accuracy"
    sparse_value = validate(sparse_path, sparse_metric, sparse_native)

    evidence = {
        "schema": "saebase.m3.native-saebench.v1",
        "upstream_commit": COMMIT, "model_name": MODEL,
        "sae_release": RELEASE, "sae_id": SAE_ID, "dataset": DATASET,
        "native_metrics": {**core_metrics, sparse_metric: sparse_value},
        "core_config": core_native["eval_config"],
        "sparse_probing_config": sparse_native["eval_config"],
        "core_file": str(core_path.relative_to(output)),
        "sparse_probing_file": str(sparse_path.relative_to(output)),
        "status": "passed",
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))

if __name__ == "__main__":
    main()
