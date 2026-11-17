# RSS 临时 case study

固定 target，在其前面分别加五段公开 demo 的 FACTS，观察 RSS 共同响应特征和变体特有特征。使用 GPT-2 Small、`gpt2small_res_jb_l8`、`blocks.8.hook_resid_pre`，在最后一个目标 token 提取完整 SAE codes。

从 SAEBase 根目录运行：

```bash
python scripts/RSS/run_case_study.py
```

默认运行全部 24 个公开 demo prompts，输出到 `experiment/RSS/<UTC时间>/`。`--output <新目录>` 指定保存位置，`--device cpu` 使用 CPU，`--limit 1` 仅看第一个案例。目录须尚未存在，以保留旧结果。

- `index.md`：所有案例入口。
- `case_XX/rss_case.md`：RSS 和变体特有特征两张表，含五个真实变化值和 Neuronpedia 链接。
- `case_XX/rss_case.json`：prompts、token IDs、模型与 SAE 身份、参数、选集、激活值和解释。
- `case_XX/rss_activations.npz`：`clean`（F）、`variants`（5×F）、`delta`（5×F）的完整原始 codes 与变化矩阵。
- `neuronpedia.json`：所有选中特征的公开解释及实际获取状态。
- `run.json`：运行参数、来源和完成状态。

模型采用公开 demo 的 `HookedTransformer.from_pretrained_no_processing("gpt2", center_writing_weights=True)`，SAE 加载、激活提取和解释复用 SAEBase API；实际加载设置写入结果。

RSS 直接使用公开仓库 `rss.py`，默认 `M=400, K=20, rho=kappa=0.8, tau=1e-5`。`source.json` 保存上游文件身份，`LICENSE` 为上游 Apache 2.0 许可证。`inputs.py` 的 prompts 和 FACTS 来自同仓库的 `assets/demo/demo.py`。

变体特有规则：从该变体绝对变化前 100 名中，取变化 > 1e-5 且至少为其他四个变体最大绝对变化五倍的特征，排除 RSS support，每个变体最多三个。没有符合条件的特征时保留空集。五倍规则只用于案例展示。

表格按真实筛选结果输出，Neuronpedia 解释保留英文原文。对目标或新增上下文的语义归属需要阅读结果判断。所有案例均保存；激活先落盘，解释获取失败也可继续查看真实变化。

## 第二轮：五个鲜明上下文

```bash
python scripts/RSS/run_contrast.py
python scripts/RSS/run_contrast.py --sae-key gpt2small_fs_24576_l8
```

先跑香蕉面包，再运行其余 23 个公开 demo prompts；支持 `--limit 1` 和 `--output <新目录>`。固定篮球、宇航员、小提琴、国际象棋、非洲象五段上下文，通过新接入的 `CaptureTask` / `mode=rss` 执行。模型仍采用公开 demo 的 GPT-2 加载设置。上下文数字候选遍历完整字典，要求 recurrence < 0.8、当前正变化 > tau、至少五倍其他变体最大绝对变化，按当前变化幅度保留全部，再获取解释。空集照实保存。

本轮原 SAE 与 Feature Splitting SAE 各保存 24 个案例，目录分别为 `experiment/RSS/20261116_distinct_contexts` 和 `experiment/RSS/20261116_distinct_contexts_fs24576`。真实两张表及案例选择结果在前者 `contrast_report.md`。该报告明确记录：香蕉面包在 Feature Splitting SAE 中出现三条目标相关、双向变化的 RSS 特征，但尚未找到同一个案例中的五个合格上下文特征。

从已有结果重建本轮阅读报告：

```bash
python scripts/RSS/review_contrast.py \
  --original experiment/RSS/20261116_distinct_contexts \
  --alternative experiment/RSS/20261116_distinct_contexts_fs24576
```

`explanation_review.json` 的关键词仅用于导航阅读；RSS 选集完全由原选择器根据激活确定。`case_selection.json` 记录展示了哪些真实行、缺少什么。报告不会把预期解释匹配当成观测结果。

## 已找到完整候选：Llama 3.1 8B，layer 21

最终真实同组结果在 `experiment/RSS/20261117_llama31_8b_l21_complete_case/case_report.md`，展示3条目标相关、双向变化 RSS 特征，以及5条各自对应新增背景、recurrence=0.2 的 RSS-excluded 特征。完整输入和8条展示行另存 `selected_case.json`；全部 codes、20条 RSS support 和所有数值候选仍保留。

本例固定请求为 `Adapt a banana bread recipe to this setting.`，新增篮球、太空、古典音乐、国际象棋、非洲象背景。目标要求根据 setting 改写配方，属于与背景有关的请求，不应将其描述为第一轮无关事实实验。

复现（`--output` 使用新目录）：

```bash
python scripts/RSS/run_contrast.py \
  --model-key llama31_8b --sae-key llama31_8b_lxr_8x --layer 21 \
  --prompts-json scripts/RSS/final_case_prompts.json \
  --contexts-json scripts/RSS/final_case_contexts.json --limit 1 \
  --output experiment/RSS/llama_complete_case_rerun
python scripts/RSS/export_final_case.py experiment/RSS/llama_complete_case_rerun
```

`search_llama.py` 是本次在16、21层探索背景和请求表达的临时搜索脚本；全部96个探索案例保存在 `experiment/RSS/20261117_llama_prompt_search`。最终完整五变体组经重新捕获生成，所有展示行均从同组结果读取。`run_contrast.py` 同时支持 `--model-key`、`--layer`、`--prompts-json` 和 `--contexts-json`，输入文件均在当前目录中可查。

## 八个指定literal-input定性案例

完整报告：`experiment/RSS/20261117_llama31_8b_eight_case_batch/summary.md`。八组原始输入位于 `qualitative_batch_inputs.json`，实际请求列表位于 `qualitative_batch_prompts.json`，逐项AI阅读判断位于 `qualitative_batch_review.json`。

```bash
python scripts/RSS/run_contrast.py \
  --model-key llama31_8b --sae-key llama31_8b_lxr_8x --layer 21 \
  --prompts-json scripts/RSS/qualitative_batch_prompts.json \
  --contexts-json scripts/RSS/final_case_contexts.json --limit 8 \
  --output experiment/RSS/eight_case_batch_rerun
python scripts/RSS/batch_candidates.py experiment/RSS/eight_case_batch_rerun
python scripts/RSS/build_qualitative_batch.py experiment/RSS/eight_case_batch_rerun
```

`batch_candidates.py` 从完整codes导出全字典RSS统计与全部符合五倍数值条件的未选中特征，保存具体排除原因，在确定ID后读取解释。它保留recurrence合格但因其他条件被排除的候选，方便按本批要求逐项判断。`build_qualitative_batch.py` 将原始输入、真实特征和明确记录的AI阅读判断合成每例一张表及八例比较摘要。原始20条RSS和所有数值候选保存在每例的 `all_candidates.md` 和JSON中，缺少清楚语义匹配时明确显示缺口。

本批8例均已实际执行，48条输入均在最后句号（token ID 13）捕获，146个不同特征解释全部取得。固定输入下尚未找到完整满足三项展示条件的案例。C02有比赛得分、太空两项背景匹配，C06有太空一项；其余为0。C06、C01、C02作为优先人工复核线索，最终是否使用由作者查看结果后决定，本次没有起草最终rebuttal。

## E1–E4诊断案例

四个诊断实验的完整输入（保留附件内部换行）在 `diagnostic_inputs.json`。结果在 `experiment/RSS/20261117_llama31_8b_diagnostics_e1_e4`，`comparison.md` 比较四组，E1–E4目录各有 `case_report.md`，完整展示全部20条RSS：clean值、五个delta、recurrence、方向、原解释与URL，随后讨论最多三条候选关系。

```bash
python scripts/RSS/run_diagnostics.py --output experiment/RSS/diagnostics_rerun
python scripts/RSS/build_diagnostics.py experiment/RSS/diagnostics_rerun
```

`diagnostic_review.json` 为全部四例完成后的AI定性阅读记录。脚本不把增加当作更优、不把无关标签拼成机制转移，也不生成模型答案或开展定量评测。保留literal换行意味着目标问题排版位置在clean和variants中并非完全一致，报告明确说明此范围。

本轮四份报告ZIP：`experiment/RSS/rss_diagnostic_four_reports.zip`，仅含E1–E4的Markdown报告。完整raw codes、delta、RSS选集、原解释、token IDs、输入与配置均留在实验目录。
