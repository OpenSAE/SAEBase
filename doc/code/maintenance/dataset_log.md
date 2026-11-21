# Datasets and source of tips

## Purpose

Describes the correspondence between the current Dataset registration and the existing PromptSource input, which helps to select the actual data and configuration.

## Scope

For data access, sample preparation and task input selection. True options are maintained by `datasets/plugins.py` and `conf/datasets/`.

## Dataset entrance

Current Dataset Registration includes: HotpotQA,BBH,CounterFact,GreaterThan,IOI,MMLUPro,SimpleCopying,RAVEL,ARCChallenge,GSM8K,HumanEval,TruthfulQA,HarmBench and XSTest.Task Read Different Dataset The sample, division and source of the declaration. `datasets/AGENTS.md`,Configure with actual data paths.

## Existing Promptsource Interface

The following is the name of the source of the hint and the input use; where the protocol number corresponds to the original data selection record. Datat and PromptSource each selects the actual registration interface.

| Behavior | Hydra group | Notes |
|----------|-------------|-------|
| SimpleCopying | `simple_copying` | Sequence / induction-style copying |
| GreaterThan | `greater_than` | Numeric / comparative relation |
| IOI | `ioi` | Indirect Object Identification |
| SubjectVerbAgreement | `subject_verb_agreement` | Morphosyntactic agreement |
| CounterFact | `counterfact` | Factual association probes |
| MMLUPro | `mmlu_pro` | TIGER-Lab MMLU-Pro MCQ (0-shot direct by default) |
| BBH | `bbh` | Big-Bench Hard; protocol 05 / 10 |
| ARCChallenge | `arc_challenge` | ARC-Challenge; protocol 05 test IDs |
| MedQA | `medqa` | MedQA-USMLE; protocol 05 test IDs |
| MATH500 | `math500` | MATH-500；`max_tokens=2048` |
| GPQAExtended | `gpqa_extended` | GPQA Extended;gated protocol 10 marks blocked |
| LiveCodeBench | `livecodebench` | Agreements 10:05 It's... 350 + 150(seed `20260913`) |

Artifacts live under `{paths.data_dir}/{Behavior}/{model_key}/v1/`.
