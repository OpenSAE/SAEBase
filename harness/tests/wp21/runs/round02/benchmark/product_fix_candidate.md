# F04 产品修复候选（未应用到冻结基线或本 Harness 分支）

观察（真实执行，见同目录 `f04_result.json`）：

1. `load_registered(..., device="cuda")` 时参数与 activation 在 `cuda:0`。
2. `load_registered` 默认 `device="cpu"`。`saebase/benchmarks/runner.py` 与 `tasks/benchmark/backend/runner.py` 调用时不传 `device`。
3. 两条不同长度输入（7 vs 32 tokens）上，`BatchEngine(pad_token_id=None)` 抛出 `mixed token lengths require explicit pad_token_id`。
4. 同样输入传入 `pad_token_id=0`（Gemma tokenizer 的 pad）后 `measured=2`，运行成功。
5. Gemma tokenizer：`pad_token=<pad>`，`pad_token_id=0`，`padding_side=left`。M2 内部使用右填充。

建议最小产品修复（另开产品 PR，不要改 Harness 评分）：

- `built_in_or_external` 把 config/options 中的 device 转给 `load_registered`。
- 构造 `BatchEngine` 时传入 tokenizer `pad_token_id`（或 `eos_token_id` 回退）。
- 两处 runner 同步修改。
