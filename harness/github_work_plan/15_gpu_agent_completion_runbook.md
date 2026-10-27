# WP15 — GPU Agent Command Handoff

**Mandatory reading:** [WP15 full decisions](15_complete_six_workflows_and_release.md). This is an action runbook, not a plan to test whether packages happen to be installed.

## Start and ownership

Start from current WP13 branch in a fresh worktree. Save the actual source SHA. Inspect nvidia-smi, free scratch, current venvs and cached assets. Preserve G1's RTX 5070 real capture and old LIVE-02/03 results; no GPU-00 rerun. Create isolated environments for core, SAEBench, circuit-tracer, training, and interpretation. Do not upgrade the historically working CUDA environment in place.

Complete G1–G6 through executable code and real outputs, not only six JSON status fields. Each feature needs one executable local command, a genuine artifact, tests, a readable public recorded-results page, and a short documented example. Fix compatibility and missing assets directly. Make code commits and PR against WP13. Never upload weights, owner-only prompts or private tokens.

## G1: capture and intervene

Reuse Gemma-3-270M L5 and its exact scope-residual SAE. Test `capture_prompt_trace` + `compare_sae_edits` on one new original prompt, with a genuinely active content feature. Compare baseline, no-op, ablate and scale. The old G1 receipt's F1627 and LIVE-03's F506/F95 are different runs; keep them separate. Test a private authenticated **edit** job too, because the previous queue smoke only showed inspect jobs.

Stage reviewed LIVE-02 source-bound tokens for the browser and the two actual LIVE-03 teacher-forced intervention cards. Do not claim text generation in those cards.

## G2: generation-time steering

Use `saebase.interventions.steering.compare_steered_generation` and `GenerationSpec` (or the existing compatible cached implementation). Implement a simple real CLI if there is no public entry point. Run two prompts, each 16–32 tokens, with baseline, no-op and 2–3 sensible edit configurations at a feature actually active at the **current** generation hook site. Save generated token IDs, decoded continuations, per-step next-token logit/probability shifts, no-op equality, latency and GPU memory.

Identical greedy text is not a failure if the forward intervention genuinely changes the next-token distribution. Show the measured probability/rank movement in the UI. Try one better-selected content feature if needed, but do not engineer false language differences.

## G3: native SAEBench setup and run

The existing `saebase.wp12.benchmarks.NativePilot` demands original SAEBench SHA `8042bb3828c6340da8d12062324e92b2077c571c` at the worktree's `upstream` directory. Execute source acquisition and verify version:

```bash
cd /workspace/SAEBase-wp15
git clone https://github.com/adamkarvonen/SAEBench.git upstream
git -C upstream checkout 8042bb3828c6340da8d12062324e92b2077c571c
git -C upstream rev-parse HEAD
python -m venv /workspace/wp15/venvs/saebench
/workspace/wp15/venvs/saebench/bin/python -m pip install -e ./upstream
```

Check pinned commit constraints before installation. Ensure SAEBase itself is importable in the venv. Then execute `NativePilot(repo_root=Path.cwd(), upstream=Path.cwd()/"upstream", output=Path("/workspace/wp15/runs/G3-native"))` with `native_pilot`. The provided `saebase/tools/m3_native_saebench_smoke.py` runs genuine bounded Pythia-70M L3 SAE / AG News Core + Sparse Probing. Resolve missing public model/SAE/dataset assets instead of stopping at the import error. Return actual L0, MSE and Sparse Probing metric values and put them into the public Benchmarks readout.

## G4: circuit-tracer install, native graph and viewer

Use original `decoderesearch/circuit-tracer` source. In isolated environment install it and run its real CLI help before invoking wrapper. Select a genuinely matching supported **Gemma-2-2B + original transcoder** or **Llama-3.2-1B + original transcoder** pair from its demo. The G1 Gemma-3-270M SAE is **not compatible by default**.

Call `saebase.wp12.circuits.run_native_circuit_tracer` for one short original prompt using the installed CLI. Convert its actual frontend v1 JSON with `import_native_graph_file`; if upstream format changed, repair the converter against that actual format. No fabricated graph JSON and no untrusted raw .pt loading in the public server. Open the real graph with nodes, signed edges, error nodes, filtering, pinning, annotations and export. Test native graph-node intervention when the original asset supports it.

With RTX 5070 12 GB, first try upstream-supported CPU offload or smaller compatible demo. If real native tracing OOMs, find an available larger existing GPU for **this job alone** and report the exact requirement; paid rental requires explicit approval.

## G5: make small data, train and reload

Use existing native Sparsify import or pinned SAELens. Choose small ungated Pythia-70M and a real compatible PyTorch hook. Generate a modest versioned local operator-authored text dataset with 200–1000 distinct rows. Verify that the **native dataset loader** can open its local HF dataset format; check load_dataset versus load_from_disk and repair this integration if necessary. Use source revision identifiers reflecting actual commits or data hashes, not invented 40-hex strings.

Use `saebase.wp12.training.TrainPlan`, `run_training(..., approved=True)` for a native 50–200-step single-GPU SAE job, `register_checkpoint`, and **really reload the new SAE into SAEBase**. On real activations produce finite codes and reconstruction; save training logs/loss and show checkpoint artifact metadata on the Training page. Do not confuse a metadata registration with a proven reload. Resume is claimed only after restoring original native training state.

## G6: real local model explanation

Download full original [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model, tokenizer, config and safetensors to a private pinned snapshot. Its official safetensors file is about 988 MB; the old ~20 KB code stub cannot run. In an isolated interpretation venv with huggingface_hub:

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /workspace/wp15/saebase/assets/models/qwen2p5_05b_instruct
```

Pin and record the real revision. Build genuine `site_id, feature_id, discovery, heldout` input JSON from actual L5/L9 sparse codes, with distinct prompt groups and both activating/nonactivating saebase.examples. If necessary capture more distinct original saebase.examples. Follow exact row contract in `saebase/research_lab/backend/auto_interp.py`.

Run `saebase/tools/wp12_autointerpret.py --local-model <actual-snapshot> --examples <actual-json> --output <private-json> --device cuda`. The local backend uses original causal language-model forward and returns model-generated candidate + held-out score. If the first feature has insufficient signal, try a better supported one with an explicit recorded decision. Review and approve a grounded candidate manually before public display; show its held-out performance as diagnostic, not proof that the semantic label is correct.

## Integration and finish line

The Integration Agent runs in parallel: source-approved Prompt → Token → Feature → Intervention, saved Generate output, real SAEBench metrics, native Circuits graph, Training checkpoint record and reviewed Explain candidate. Reuse existing FastAPI/JS views; no blank feature panes. CPU Space shows saved actual results and reproducible local commands, not false hosted GPU inference.

Smoke test a populated staging Docker and real browser navigation for all six; test process restart, private route isolation, install documentation and rollback revision. Then submit code PRs, lightweight receipts, small permitted display assets and actual staging URL. Do **not** report production updated until it is verified.

## Required receipt

For each job produce one small `/workspace/wp15/receipts/Gx.json` with real source SHA, upstream SHA, model/SAE identities, exact executed command, actual output value(s), artifact path, elapsed time, GPU memory if relevant, working UI route and blocker (if any). Do not fabricate missing values. Update `15_release_completion_matrix.csv`: `NOT_STARTED`, `RUNNING`, `WORKING`, or `EXTERNAL_BLOCKER`. “Not installed” and “no local snapshot” are **not** final blocker states. Fix them or prove a genuinely restricted asset/hardware issue and give one usable compatible alternative.
