# WP12 Core Functionality Assessment

Date: 2026-10-22. Source branch: `feat/wp12-p3-p8-platform`.

**PASS means its named core operation has an executable SAEBase / native-upstream code route, with a bounded source/test entrypoint.** It does not mean real native/GPU parity, all model sites, deployed services, or independent P8 certification. A route requiring explicit local dependencies remains scoped to that environment.

**PARTIAL means the stated core behavior itself is missing a material operation.** Grade 2/3 = a useful executable subset but not complete; 1/3 = interface/scaffold only; 0/3 = no implementation. **BLOCKED** means the core execution cannot presently be completed without independent external hardware/access. Core PASS and P8 scientific release are different axes; all P8 entries remain PENDING.

Current implementation: **44 PASS · 4 PARTIAL · 2 BLOCKED**. Final P8 RELEASED: **0/50**.

| ID | Function | Core | Grade | Actual core operation | Execution scope or remaining core gap |
| --- | --- | --- | --- | --- | --- |
| F01 | Feature dashboard | PASS | 3/3 | Open a source-identified SAE feature card with real activation examples and provenance. | public read-only |
| F02 | Feature search | PASS | 3/3 | Search by approved explanation, recorded prompt or fresh private-worker SAE token activations. | recorded public / live private worker |
| F03 | Automatic feature interpretation | PARTIAL | 2/3 | Generate descriptions from actual discovery examples and score on disjoint held-out groups. | Connect a usable local/model-provider generator and candidate review/publication UI. |
| F04 | Feature steering | PASS | 3/3 | Compare original and steered next-token generation from the same prompt with real feature edits. | local model |
| F05 | Circuit exploration | PASS | 3/3 | Open, inspect, prune and annotate a source-bound attribution graph. | browser saved graphs |
| F06 | Model loading | PASS | 3/3 | Load registered TransformerLens-backed models and retain model processing identity. | local TL3/TL4 adapter |
| F07 | Activation caching | PASS | 3/3 | Capture and reuse hash-keyed activations/codes without stale cross-model hits. | local cached model |
| F08 | Activation patching | PASS | 3/3 | Patch a real activation and verify the requested hook fires. | local model |
| F09 | Hook interventions | PASS | 3/3 | Discover and invoke supported named model hooks, rejecting nonexistent hooks. | local hooks |
| F10 | Component analysis | PARTIAL | 2/3 | Inspect real attention/MLP and residual component contributions, including individual heads. | Expose native per-attention-head and neuron/component attribution; current implementation only returns block-level residual differences. |
| F11 | Pretrained SAE loading | PASS | 3/3 | Load a real public pretrained SAE and run native encode/decode. | local SAELens |
| F12 | SAE training | PASS | 3/3 | Launch native SAELens SAE training with tracked checkpoint and resume inputs. | opt-in native runner |
| F13 | Multiple SAE architectures | PASS | 3/3 | Choose Standard, TopK, JumpReLU or Gated native SAE training architectures. | native SAELens config |
| F14 | Model integration | PASS | 3/3 | Pair an SAE with the correct model layer, hook, dimensions and explicit provenance. | local model+SAE |
| F15 | Feature analysis and visualization | PASS | 3/3 | Compute SAE features for real prompts and display source-linked analysis. | local and reviewed browser |
| F16 | Intermediate states and gradients | PASS | 3/3 | Capture actual NNsight intermediates and autograd gradients. | local NNsight opt-in |
| F17 | Activation editing | PASS | 3/3 | Perform real NNsight activation output replacements and collect readout. | local NNsight opt-in |
| F18 | Multiple input tracing | PASS | 3/3 | Trace multiple independent inputs using NNsight native invokers. | local NNsight opt-in |
| F19 | Generation-time intervention | PARTIAL | 2/3 | Intervene during native NNsight text generation at declared token steps. | Replace archived task-only generate dependencies with a standalone source-validated native NNsight per-generation-step execution path. |
| F20 | Remote NDIF execution | BLOCKED | 2/3 | Run an authorized real NDIF remote trace on a deployed supported model. | Requires available NDIF service, eligible deployed model, owner credential and permitted remote prompt transfer. |
| F21 | Activation intervention | PASS | 3/3 | Build a native pyvene IntervenableModel and execute source-to-base activation edits. | native pyvene optional |
| F22 | Composable intervention | PASS | 3/3 | Choose parallel or serial native pyvene interventions across typed components. | native pyvene optional |
| F23 | Fine-grained intervention | PASS | 3/3 | Select individual SAE features, token positions and PyTorch neuron subsets for edits. | local model/Torch |
| F24 | Cross-architecture support | PASS | 3/3 | Run real PyTorch module edits on non-Transformer models via native hooks. | local PyTorch |
| F25 | Serializable configurations | PASS | 3/3 | Save and reload intervention plans and original pyvene artifacts. | local trusted pyvene |
| F26 | Feature-level circuit discovery | PASS | 3/3 | Compute a feature attribution circuit by calling the original circuit-tracer CLI. | opt-in native CLI |
| F27 | Attribution graph construction | PASS | 3/3 | Convert native circuit-tracer JSON to a typed graph preserving signed edges and errors. | native graph import |
| F28 | Interactive graph visualizations | PASS | 3/3 | Render signed circuit nodes/edges with filtering, zoom and node inspection. | browser source-bound graphs |
| F29 | Graph grouping and annotation | PASS | 3/3 | Pin, group, annotate and export/reopen annotated graph nodes. | browser local annotations |
| F30 | Circuit interventions | PASS | 3/3 | Select a traced transcoder feature and perform a real causal model-output edit. | local single-layer transcoder |
| F31 | Feature-centric view | PASS | 3/3 | Browse an SAE feature's highest reviewed contexts and descriptions. | browser source-backed |
| F32 | Prompt-centric view | PASS | 3/3 | Load a real prompt trace and inspect selected feature activations. | recorded public/local private compute |
| F33 | Interactive token inspection | PASS | 3/3 | Select an exact token and rank genuinely active SAE features. | browser saved traces |
| F34 | Correlation analysis | PASS | 3/3 | Compute group-level feature correlations from complete nonzero SAE records. | CPU source-bound |
| F35 | Portable visual output | PASS | 3/3 | Export standalone source-escaped feature HTML without external dependencies. | browser offline HTML |
| F36 | Native standardized evaluation | PASS | 3/3 | Launch original native SAEBench Core and Sparse Probing and import untouched metrics. | opt-in native CPU/GPU |
| F37 | Reconstruction and sparsity | PASS | 3/3 | Compute and compare reconstruction MSE/sparsity under explicit source protocols. | local native/imported |
| F38 | Downstream behavioral evaluation | PASS | 3/3 | Execute downstream Sparse Probing against original train/test dataset splits. | opt-in native evaluator |
| F39 | Custom SAE integration | PASS | 3/3 | Evaluate registered custom SAEs with original SAEBench native evaluator. | opt-in native custom SAE |
| F40 | Benchmark comparison | PASS | 3/3 | Compare two genuinely matched SAE metrics along L0 with source identity guards. | local/browser matched results |
| F41 | Pretrained model collection | PASS | 3/3 | List and load published Gemma Scope 2 assets by verified source/revision. | official pinned model repository |
| F42 | Multiple activation sites | PASS | 3/3 | Resolve official model/layer/hook coordinates for SAE and transcoder inputs/outputs. | official config native |
| F43 | Transcoder models | PASS | 3/3 | Load genuine Gemma Scope 2 JumpReLU transcoders including affine skip, encode/decode. | local official safetensors |
| F44 | Cross-layer representations | PARTIAL | 2/3 | Load and intervene with native multi-layer crosscoders and cross-layer transcoders. | Implement full official CLT source-target layer contractions and native multi-layer intervention, not just compatible 3D tensor shapes. |
| F45 | Width and sparsity variants | PASS | 3/3 | Enumerate and load real published width, L0 and optional affine variants. | official pinned manifest |
| F46 | TopK training | PASS | 3/3 | Run native Sparsify TopK training with real model/data and checkpoint saving. | opt-in native training |
| F47 | On-the-fly activation extraction | PASS | 3/3 | Train from model activations extracted on the fly via Sparsify native hooks. | opt-in Sparsify on-the-fly |
| F48 | SAE and transcoder training | PASS | 3/3 | Choose genuine native SAE versus transcoder training mode. | opt-in Sparsify native |
| F49 | Custom hookpoints | PASS | 3/3 | Choose and validate native Sparsify training hookpoints instead of fixed layer outputs. | opt-in native custom hook |
| F50 | Distributed training and checkpoint reuse | BLOCKED | 2/3 | Train and resume genuine multi-GPU distributed SAE with preserved optimizer state. | Requires a real 2+-GPU allocation and an executed distributed restart/resume, not a simulated single-rank run. |

## What core PASS does not authorize

Native TL4 model parity, new hosted GPU endpoint, paid external APIs, original dataset redistribution, arbitrary pyvene/NNsight/circuit model combinations, full SAEBench suites, actual two-GPU distributed training, and independent researcher testing are separate follow-up gates. They are **not** converted to PASS simply because one adapted route can run. In particular F20 and F50 are blocked from actual external execution.

## Core gaps to implement next

- **F03:** Connect a usable local/model-provider generator and candidate review/publication UI.
- **F10:** Expose native per-attention-head and neuron/component attribution; current implementation only returns block-level residual differences.
- **F19:** Replace archived task-only generate dependencies with a standalone source-validated native NNsight per-generation-step execution path.
- **F44:** Implement full official CLT source-target layer contractions and native multi-layer intervention, not just compatible 3D tensor shapes.
- **F20:** Requires available NDIF service, eligible deployed model, owner credential and permitted remote prompt transfer.
- **F50:** Requires a real 2+-GPU allocation and an executed distributed restart/resume, not a simulated single-rank run.

## Source traceability

Each row in `12_capability_acceptance_matrix.csv` contains its supported entrypoint, narrow capability scope, `core_test`, specific implementation gap if present, `core_grade`, and an independent `p8_status`. A PASS may have only CPU source-contract tests; it does not replace native numerical evidence. Avoid promoting `p8_status` without real evidence and review.
