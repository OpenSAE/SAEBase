# SAR threshold for primary active density

## Background and objectives

Use the same 100-track MMLU-Pro rerun 32 candidate SAR to measure the interaction of characteristics that precludes a strictly primary active density of more than 1%. Use the existing density threshold parameters, add a comparison of the boundary switches, and use the established rules that contain the equivalent when omitting the switches.

## Implementation record

- Checks borders with a complete Neuronpedia density snapshot. The density is just 1%. The characteristic is 19174, the other exclusion rule.
- Retains the order, model, SAE, reference set, filtering and regression settings of the original 100 questions. The official CLI command, effective configuration, console and completion list is located in [this operating directory](../../../../experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density1pct).
- Use the original run-by to activate the re-election of the old candidate, 100 questions are fully consistent with the saved Top 32. The expected number of candidates for the strict threshold is stored in this run-in directory to check the official results on a question-by-issue basis.
- Configure passes and skips with 61 of the SAR tests. Ruff and configuration documents are checked.
- Starts the automatic pre-check and full run in tmux session `saebase-sar-density1pct`. The session and exit code file are in [starting record](../../../../experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density1pct/launch.json).
- Both questions are pre-checked for automatic performance. The official running exit code is 0, completing the 100/100 question, and the failure, undefined and blocked are 0. The performance report records a wall clock time of about 18 minutes.
- Checks the list by formally reading the entry. 100 tips and models, SAEs, data identities and controls are consistent, and all new candidates are in line with the forecast. Checking 100 original arrays of SHA-256 on a question-by-issue basis, save 1,000 Top 10 features and 2,000 end-point comments.
- The density rule, stricter than 1%, excludes 399 features, with 26,857 participation in the Whiten ranking after a combination of filters. Compared to 50% of thorium, the candidate for 100 questions has changed and replaced 2,415 positions. The results are explained by the regression factor for each of their own candidate pools and intervention combinations. [Comparative report](../../../../experiment/MMLUPro/llama31_8b/capture/sar/top32_candidates/density1pct/reports/density_comparison.md)]
