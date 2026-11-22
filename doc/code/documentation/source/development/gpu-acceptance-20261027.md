# SAEBASE GPU Receiving and Inspection Report

Date: 2026-10-27. Directory of evidence: `/workspace/SAEBase-gpu-acceptance-20261027`. This report records only the results of the run and the documents left.

## Conclusions

Seven. SAE It's all here. RTX 5070(12GB)Use real. Gemma-2-2B . This is a running smoke and limited validation, not end-to-end scientific acceptance.. Capture,Intervene,Generate,Benchmark The numbers are from the official mission entrance.. Circuits In a quarantine. `circuit-tracer 0.5.0` I wrote it. 118 Nodes, 3636 The original map of the strips, the fusion and the characterization of which are still pending.. Explain It's... `heldout_score` It's just... SAE Activate consistency, not semantic correctness. Train Only completed 2 Step.`tasks.generate.backend.generate` I'm sorry. vLLM 0.27.1 Generated 8 One. token And move to the baseline one by one. token Same.

The first round of repairs is at `f7dca13`. This round of functional submissions is `d866172`, followed by one submission that only changes the Ruff layout. There is no consolidation of PR, no distribution, no changes to `.cursor/` or approval rules, and no research nodes have been marked as validated.

## Environment

| Item | Value |
| --- | --- |
| Checkout | `/workspace/SAEBase-acceptance`, branch `sync-openresearch-harness-20261027` |
| HEAD | `548d61a9d9f08cbc7b575d8982b37ff70a426426` |
| GPU | NVIDIA GeForce RTX 5070, 12227 MiB, computing capability 12.0 |
| Driver | 595.91.07 |
| Main environment | Python 3.12.14，`/venv/main`，torch 2.11.0+cu128，CUDA 12.8 |
| SAE Inn | transformer-lens 3.9.0，sae-lens 6.54.12，datasets 5.1.0 |
| Model | `google/gemma-2-2b`,Snapshot. `c5ebcd40d208330abc697524c919956e692655cf` |
| SAE | `google/gemma-scope-2b-pt-res`,`layer_25/width_16k/average_l0_116`,HF Revision `fd571b47c1c64851e9b1989792367b9babb4af63` |
| Post-load status | Model sha256 `a3aa44f999c0869e4dd34faeb9ca4674e4dd838fe97a6ad74f5de0b22477677f`,SAE sha256 `61f521f1c188093c2091e4d1834f0e78d59d50deb7922992dcfa66c9b58cc8b9` |
| The probe. | `logs/environment_probe.json` |

The hint is always `The capital of France is`. Sha256 `bbaff4d2ecd5892d4a442b0f53131641bf6e6f284761dd20fc0664bc97145762`. Hooked Transformer's token id is `[2, 651, 6037, 576, 6081, 603]` (including BOS).

Hugging Face reads the token provided by the current session, which is only in the process environment. Token is not included in this report, in the warehouse or in the evidence JSON.

## Seven missions.

### Capture: Through

The public entry point is executed after the `train.log` of Hydra has been rewritten into the acceptance log. Result: `experiment/IOI/gemma2_2b_base/capture/gpuacc20261027_42/metadata/capture_result.json`.

- Status `executed`,Bits `blocks.25.hook_resid_post`
- Characteristics 16384, activated 98
- Max. 14325, activate 1711050702148438.
- Reconstruct L2 514.810302734375

On the first round of unreformed log paths, the empty `train.log` will be entered in the catalogue of experiments by Hydra, `guard_existing_run`, followed by `FileExistsError`. Log: `logs/capture_unmodified_entry.log`. At that time, it was bypassed by `hydra.job_logging.handlers.file.filename`. D1 has removed this bypass, see rechecking below.

### Intervene: Through

`experiment/IOI/gemma2_2b_base/intervene/gpuacc20261027_42/metadata/intervene_result.json`.Features 14325 Set 0.`logit_delta_l2` 431.2317810058594,`max_abs_logit_change` 3.328125.Edit Before and After top-5 token Both. 476,573,974,1170,2078.Identity patch in 1e-4 I've got a pass through..

Same tip. `evaluate_edit` / `apply_condition` It's... L2(`extra/operator_matrix.json`):

| Operation | logit L2 |
| --- | --- |
| Constant level (empty replacement) | 0.0 |
| Melt. | 431.2317810058594 |
| Zoom in x 2 | 446.2266845703125 |
| Add 80 | 206.68472290039062 |
| Catch 200. | 75.37415313720703 |

Melting is the same as official intervene.

### General: Pass

`experiment/IOI/gemma2_2b_base/generate/gpuacc20261027_42/metadata/generate_result.json`.Features 14325 Add 80,`max_new_tokens=8`,temperature 0,seed 42.Backend is... `compare_steered_generation` No, no, no. vLLM,No, not at all. KV cache.

- Baseline token `[476, 3413, 576, 82777, 235265, 1165, 603, 476]`, text ` a city of contrasts. It is a`
- Turn token`[476, 3413, 674, 603, 2247, 576, 4281, 578]`, text` a city that is full of history and`
- The first of the different subscripts is 2

### Circuits: Pass

`task.approved=false` Yes. `PermissionError`(`logs/circuits_gate.log`).Not in the main environment after approval. `circuit-tracer`.Isolate virtual environments `/workspace/venvs/circuit-tracer`(`--system-site-packages`,Continue torch 2.11+cu128,No downgrade main environment) installed `circuit-tracer==0.5.0`.In the environment. `hf_utils.py` Yes. `huggingface_hub` 1.x Get rid of it. `HF_HUB_ENABLE_HF_TRANSFER` . This changes only the installed copy in the virtual environment, and it does not. SAEBase Warehouse.

`tasks/circuits/operation.py` Now turn it on. `offload`,`dtype`,`batch_size`,`max_feature_nodes`,`lazy_encoder` To what already exists. `run_native_circuit_tracer`.This Call:`transcoder_set=gemma`(Parsing to `mwhanna/gemma-scope-transcoders`,Revision `bd5773156dea09893636c801df1237d0410307d2`),float16,offload cpu,batch 1,Most 128 Characteristic Node, lazy encoder.

Results: `circuits_cwd/figures/circuit_trace/native_graph.pt`, 480017 bytes, Sha256 `6ab377abe0f7a099deca9f4e35575e41fffe4c49dc84173404edd6d80964a8f7`. The upstream log indicates the used Nodes 118, used Edges 3636. Status `native_raw_graph_generated_not_normalized`.

### Explain: Pass

The Hydra entrance cannot be accessed; `execute(cfg, generator=None)` will refuse to create an explanation. Supported entry will be `generator` to `execute` for loaded models. The script `logs/explain_execute.log` and the result `extra/explain_execute.json`.

The generator is... `google/gemma-2-2b` It's... `HookedTransformer.generate`,temperature 0,24 New token.Synchronising folder:

> Paris. The capital of the United States is Washington, D.C. The capital of the United Kingdom is London.

Status `heldout_tested`, `heldout_score`1.0. This fraction indicates only whether or not the projection is consistent with the same SAE activation and is not a separate synonym. The collections are g-france (171.105) and g-quatum (0), leaving the collections g-paris (65.944) and g-nums (0).

### Benchmark: Pass

`experiment/IOI/gemma2_2b_base/benchmark/gpuacc20261027_42/metadata/benchmark_result.json`，`benchmark_id` `fcdec22b62be9218605c8e1c`。

| Indicators | Value |
| --- | --- |
| core.l0 | 97.0 |
| core.reconstruction_mse | 25.435324005031127 |
| core.explained_variance | 0.826606639166795 |
| reconstruction.loss_recovered | 0.9797510631328847 |
| Melting logit L2 | 430.14892578125 |
| Add 80 logit L2 | 205.2344512939453 |

Capture's active number is 98, Benchmark's L0 is 97. The melting L2 and `evaluate_edit`'s 431.231781005894 is close but not the same. Both sets come from real forwards and report in parallel and do not describe them as the same amount.

### Train: Pass (2 walk smoke)

Budget: Layer 0, `blocks.0.hook_resid_post`, `d_sae=64`, `k=8`, Topk, step 2, Batch 16, context 16,32 local text. `task.approved=false` will be rejected and `Native SAE training requires task.approved=true` will appear in the log.

Gemma-2-2B of float32 is OOM on 12GB (approximately 11.46 GiB). `LanguageModelSAERunnerConfig.dtype` will not pass to HookedTransformer. In the first round, the model is loaded into fixed float16. D2 is changed to visible `model_dtype` and `sae_dtype`, and the default is float32; the 12GB must be uploaded to `model_dtype=float16`. `save_to_disk` language is `load_from_disk`. The snapshot directory name is shown to `google/gemma-2-2b`.

Results: `experiment/IOI/gemma2_2b_base/train/gpuacc_smoke_42/native_sae_training/checkpoint/inference/sae_weights.safetensors` (1189416 bytes). `SAE.load_from_disk` Got `TopKSAE`, `d_in=2304`, `d_sae=64`, `W_enc` Shape `[2304, 64]`. Randomly entered encode absolute value and 0.5275761485099792, reconstruct MSE 1.014569345474243. Record: `extra/train_reload.json`. Timed 1578 seconds. This is a smoke-smoke, not a layer 25.

## vLM Generation

Main environment `vllm` Just about. 30 One. Python File, Missing `logging_utils`,Cannot import. No reloading to main environment to avoid replacement cu128 It's... torch.

Using `/workspace/SAEReasoning/.venv` already installed vLLLM 0.27.1 and toch 2.13.0+cu130, `PYTHONPATH` points to this SAEBASe check-out, calling `tasks.generate.backend.generate`. Gemma-2 rejects fload16 (value unstable) and replaces bflota16. `max_model_len=256`, `gpu_memory_utilization=0.80`, feed42. Scripts must be real: vLLM subprocesses will not be found `/workspace/<stdin>` from stdin.

The result `extra/vllm_generate.json`:8 new token `[476, 3413, 576, 82777, 235265, 1165, 603, 476]`, text ` a city of contrasts. It is a`. Same as the non-direction baseline of the Generate mission token. The length of the vLLM prompt is 5, with no Hooked Transformer BOS id 2.

## Data

`saebase.datasets` is exactly the same as IOI, GreaterThan, SimpleCopying, SubprojectVerbAgreement, generated under `saebase.datasets.NLP`. `extra/dataset_prompt_identity.json`. The source file diff is mainly a line change for the IOI name list and produces the same result.

CounterFact Normalize in `azhx/counterfact` Revision `c01c413f856ee38f5c080c9fc5e87aff478e2ff9` Before 20 Lines are consistent. 19728 Okay..

MMLU-Pro`TIGER-Lab/MMLU-Pro` Amend `b189ec765aa7ed75c8acfea42df31fdae71f97be`, column 12032.

BBH `lukaemon/bbh` must be specified config. This time at `boolean_expressions`, amend `982bb89fd79532a8ac676a61fc42eb1aeec63f99`, column 250. `extra/dataset_bbh.json`.

## B2

rcline remote `research-5090`, type b2. Do `copy` (not sync) with `harness/transfer/b2.py`, target `research-5090:research-5090/saebase-gpu-acceptance-20261027`. The load is two small files without documents. Then `verify` (`rclone check --one-way`): 0 documents, 2 watching files. No other objects in the barrel have been removed.

## CLI and Atlas

The source code `saebase --help` has complete parameters, and `--list` lists the task `sae` and Gemma-2-2B models. `python -m saebase.runtime.train --cfg job --help` lists the daset, model, task (including capture to train).

V0.5.0 wheel released does not contain abundled research runtime. `wheel/opensae_saebase-0.5.0-py3-none-any.whl` re-engineered by this branch consists of `saebase/runtime/train.py`, `conf/sae_research.yaml` and `datasets/plugins.py`. `scripts/distribution.py` returns after installation outside the checkout directory, three CLIs, package resources and external working directory are OK and implemented `python -m saebase.runtime.train --cfg job --help`. PyPI, public `saebase` commands have not been uploaded.

Atlas: `harness.code.build` and `harness.visualization.export` write to the receipt and inspection directory `atlas/` without changing the harness page in the warehouse. `build_pages --check` returns 0 to the existing warehouse page. Export count: core 26, knowledge 29, code 1857, environment 15, const.64. These are the export of existing capabilities and do not mark nodes as verified.

## with v0.5.0

`saebase/api/features.py`,`saebase/api/model.py`,`saebase/api/__init__.py`,`saebase/sae/loading.py`,`models/hooked_lm_common.py` With Tabs v0.5.0 Bytes Same. Capture And edit numbers is the set of numbers that are achieved. GPU Run the same code again..

## Entered PR#60 code

`f7dca13` contains the first round of two on-the-job repairs and the first draft of this report:

1. `tasks/train/backend/training.py`: The snapshot catalogue map is an official model id; local `save_to_disk`/ JSON language.
2. `tasks/circuits/operation.py`: Transfer the task profile offload, dtype, bat, feature caps and lazy encoder to the existing curcuit-tracer package. Default not to fill in must still be true. `task.approved`

This round amends the Hydra default log, the training dtype contract, the order of verification of the Circuits parameters, the Explain white list generator, and the wheel pack range. The corresponding tests are `tests/test_hydra_first_run.py`, `tests/test_train_plan_contract.py`, `tests/test_circuits_passthrough.py`, `tests/test_explain_generator.py`.

## D1–D6

| Item | Result | Evidence |
| --- | --- | --- |
| D1 Hydra first run | PASS | No longer rewriting the log outside the directory. `python -m saebase.runtime.train experiment.sub_name=d1capture seed=43 task.execute=true task.prompt='The capital of France is' task.device=cuda`, exit code 0. There is no `train.log` in the leaves. `metadata/capture_result.json` is still the active number 98, feature 14325, activation 17110502148438, re-constructing L2,514.810302734375. The CPU returns over-covered clean directory, same id rejected, another seed, and maintains `metadata/run.json` after failure. |
| D2 training dtype and data path | PASS (2 walk smokes) | `model_dtype=float16`,`sae_dtype=float32` Visible entry. New directory `train/gpuacc_d2_42/native_sae_training`.`SAE.load_from_disk`:`W_enc` `[2304, 64]` float32,Limited, absolute and 27.660303115844727.Time-consuming 10.25 sec. Unauthorized training is denied in single tests. Old plan Missing dtype Press on field float32 Comparison. |
| D3 Circuit parameters and integration | PASS (parameters); BLONKED (consolidation) | Illegal offload, dtype, nodes, batts, timeout refused before creating the directory. `approved=false` is `PermissionError`. The real front end of JSON is `0_1_2`, and the official importer requests `(\d+)_-1_(\d+)` to drop `Native reconstruction-error node mismatch`. No guess id, no feature intervention. |
| D4 Explain Generator | PASS (smoking while running); scientific findings to be determined | Only the white list `saebase_generate`, `os.system` was rejected. The VLM returned text `Paris. The capital of the United States`, 8 new token. `extra/d4_saebase_generate.json`. This is not semantic PASS. The main environment vLLM is still incomplete and this path cannot be started in the official interpreter. |
| D5 wheel | PASS (Build in this branch); published v0.5.0 without new research running time | See above CLI Paragraph.`logs/wheel_smoke.log`. |
| D6 Reporting and the environment | PASS | The main environment torch/vLLM is not reloaded. The word “not submitted” above has been changed to `f7dca13` with the current round difference. |

D7: This machine `python -m pytest -q tests` is 437 passed, 13 Skipped. `python .cursor/hooks/validate_repo.py --full` is PASS. The term Python 3.11 on GitHub and 3.12 offline-tests, Hydra, Package, Harness-source has passed. First Quality has failed because of Ruff format, and rewinded. The machine does not have Python 3.11. The words of the no-name image task in the work plan triggers a warehouse-prohibiting test, but only the phrase has been changed without changing the approval, budget or Harness rules.

## Remaining gaps

- The original chart of the circuit cannot be transformed without prejudice into a standard map of SAEBASE, and the feature intervention was not implemented.
- Explain No Independent Semantic Label.`heldout_score=1.0` Only if the activation occurs..
- Train is still a 2-step smoker on layer zero.
- The main environment vLLM is incomplete. Available engines are at `/workspace/SAEReasoning/.venv`, vLLM 0.27.1.
- V0.5.0 wheel published does not contain abundled research runtime. The newly constructed wheel of this branch contains, but does not upload, PyPI.
