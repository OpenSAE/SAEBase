# conf/datasets_log.md — prompt sources

| Behavior | Hydra group | Notes |
|----------|-------------|-------|
| SimpleCopying | `simple_copying` | Sequence / induction-style copying |
| GreaterThan | `greater_than` | Numeric / comparative relation |
| IOI | `ioi` | Indirect Object Identification |
| SubjectVerbAgreement | `subject_verb_agreement` | Morphosyntactic agreement |
| CounterFact | `counterfact` | Factual association probes |
| MMLUPro | `mmlu_pro` | TIGER-Lab MMLU-Pro MCQ (0-shot direct by default) |
| BBH | `bbh` | Big-Bench Hard；协议 05 / 10 题单 |
| ARCChallenge | `arc_challenge` | ARC-Challenge；协议 05 测试 IDs |
| MedQA | `medqa` | MedQA-USMLE；协议 05 测试 IDs |
| MATH500 | `math500` | MATH-500；`max_tokens=2048` |
| GPQAExtended | `gpqa_extended` | GPQA Extended；gated 时协议 10 记 blocked |
| LiveCodeBench | `livecodebench` | 协议 10：05 的 350 + 150（seed `20260913`） |

Artifacts live under `{paths.data_dir}/{Behavior}/{model_key}/v1/`.
