# LLM SAE research methods and verification boundaries

This document provides terminology and evidence for the LLM SAE methods represented in the Atlas Knowledge Tree. Each experiment selects its dataset, model, and task explicitly. The source contains interfaces for Gemma-2 2B, SAE loading, activation capture, interventions, generation, circuits, explanations, benchmarking, and training.

## Research questions and implemented interfaces

- **Feature capture:** Obtain activations from a specified model, layer, and hook, then encode them into SAE features. Evidence: `saebase/api/features.py`, `tasks/capture/operation.py`.
- **Teacher-forced intervention:** Replace activations or SAE features while holding the input and hook fixed, then compare downstream outputs. Evidence: `tasks/intervene/operation.py`.
- **Generation intervention:** Apply feature changes during autoregressive generation and compare baseline and edited continuations. Evidence: `tasks/generate/operation.py`.
- **Circuit analysis:** Investigate candidate causal paths using source-identified model components and intervention tracing interfaces. Evidence: `tasks/circuits/backend/circuits.py`.
- **Automatic explanation:** Propose feature explanations from examples and test them on separate data. Evidence: `tasks/explain/interpretation.py`.
- **SAE benchmarking:** Run reconstruction, interventions, controlled comparisons, reliability checks, and SAR/RSS adapters. Evidence: `tasks/benchmark/backend/runner.py`, `tasks/benchmark/integrations/sar.py`, `tasks/benchmark/integrations/rss.py`.
- **SAE training:** Train sparse autoencoders through the native training interfaces. Evidence: `tasks/train/backend/training.py`.
- **Models and prompts:** Identify model checkpoints and traceable prompt artifacts. Evidence: `models/gemma2_2b_base/gemma2_2b_base.py`, `datasets/artifacts.py`.

## Meaning of research status

An SAE method node marked **implemented** means that its specifically named code operation or data protocol exists in the source. It does not establish causal correctness of an interpretation, the quality of generation steering, benchmark superiority, or successful convergence.

Research questions needing independent experiments retain their **partial** status. Empirical claims require separate model identities, SAE checkpoint identities, dataset splits, controls, measured outcomes, and reproducible artifacts.

The Knowledge, Code, and Core trees represent scientific questions, implemented source facts, and project design respectively. Their roles remain separate even when they refer to the same research topic.
