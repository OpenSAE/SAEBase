# SAR 探索脚本与结果

目的：集中保存这次 Llama 3.1 8B、Llama-Scope L21 的 SAR 探索脚本、原始测量和逐题中文对比。范围是 GSM8K、MMLU-Pro、ARC-Challenge、TruthfulQA、BBH 各10题，沿用各自100道参考题的统计及 SAR 专属过滤。

## 正式Capture入口

筛选、g(S)与穷举交互已接入正式Capture，参数定义见[Capture说明](../../tasks/capture/README.md)。`mode=sar`对应原回归方法，`mode=I`对应穷举交互。新的运行使用`run_capture.py`，它沿用原实验的数据集配置及模型／SAE，通过正式Runtime和自动预检执行。

```bash
# 默认过滤10个候选，只输出一个删除后g最高的single
python -m scripts.SAR.run_capture --dataset GSM8K --mode g_s --filter 10 --capture-count 1
# 同一候选池，完整穷举后输出最大I的pair
python -m scripts.SAR.run_capture --dataset GSM8K --mode I --filter 10 --capture-count 1 --tag capture_I_top1
# 复现旧表的全SAE whiten / non-whiten Top1；参考题仍为100道
python -m scripts.SAR.run_capture --dataset GSM8K --mode whiten --filter all --capture-count 1 --tag capture_whiten_all_top1
python -m scripts.SAR.run_capture --dataset GSM8K --mode non_whiten --filter all --capture-count 1 --tag capture_raw_all_top1
# 用现有50题真实cube核对正式Capture的候选、四项Top1和原始数值，不重新推理
python -m scripts.SAR.check_capture_reproduction
```

`--questions`单独控制实际题数，默认10；`--check-config`仅检查配置。运行产物使用标准`experiment/<dataset>/<model>/capture/`路径，旧`results/`报告继续保留。默认filter=10会让whiten和non-whiten也应用过滤；旧报告的这两项使用filter=all，所以复现旧表时要按上面的方式明确设置。

活跃脚本的四格排名与single排名直接调用`tasks.capture.behavior`，完整上下文尾层评分调用`tasks.capture.answer`；`pair_search.py`也转交正式Capture实现。历史源码快照仍保留在`legacy/`及`results/code/`。

## 查看结果

- `results/report_zh.md`：完整题目、答案、最大I的pair、最大g01的single、whiten与non-whiten第一名、中文解释和数值。
- `results/report_ablation_zh.md`及同名JSON：用户在计算中确认single为“仅清零一项”后的实时版本；`watch_reports.py`独立更新，完成后同步到上述最终报告。
- `results/report.json`：同一内容的结构化数据，另含基线合格的最大I、其余八项全保留时的最大I、10个single的完整排名。
- `results/<数据集>-<题号>/coalitions.npz`：1024种组合的全部实测g，含feature_ids、masks、historical标记及答案token。
- 每题的 `verification.json`、`complete.json`、`measurements.jsonl`：重放校验、结果SHA、逐组合断点与来源。
- 每题的 `source_sar.npz`：原384次SAR测量的原文件副本，按原SHA核对。
- `results/reference/`、`reference_sources.json`：五数据集原mean/std及过滤配置副本与来源SHA。
- `results/plan.json`：冻结原题、完整输入、原候选、源产物SHA、模型与SAE身份。
- `results/execution.json`、`results/exhaustive.log`：运行协议和日志。
- `results/audit.json`：用独立十维差分计算复核最大I、全部mask覆盖、原测量、single与概率的检查结果。

## 口径

对每题原先选定的10个候选测量全部1024种保留/清零状态；其他SAE特征和重构残差保持原值。对45个pair逐个枚举其余八项的256种背景，计算 `I=g11-g10-g01+g00`，取**原值最大**。不是32768维SAE全部特征对，也不是SAR二阶拟合系数。

single按用户确认的定义：仅清零一个候选，其余九项及Top10以外特征保持原样，取清零后g01最高的一项。其概率变化相对于原始全部保留的状态。结构化数据另保存最大I背景下更好的single，避免混用不同背景的数值。

whiten为全SAE最高的绝对zscore，non-whiten为全SAE最高的绝对原生activation，均不施加SAR专属过滤；报告额外保留SAR过滤后whiten第一名。zscore使用原100题所有prompt token的mean/std，std=0排除，无正数std下限。

g是标准答案teacher forcing的平均token log概率；平均token概率为exp(g)，答案联合概率为exp(Tg)，不含EOS。解释来自原SAE的Neuronpedia发布解释，中文为翻译，不能仅凭解释断言因果含义。

原384次/题有324种不同状态（含全保留）；每题补测700种。只有模型与SAE权重SHA一致、原激活与native codes一致、尾层计算和完整前向及历史g一致时才复用旧值。计算采用原BF16模型、FP32 SAE、FP32 log_softmax和FP64均值。前22层在GPU执行并卸载权重到CPU，缓存完整序列的中间激活；每个mask计算完整上下文的尾十层，没有使用KV近似或代理模型填值。

仅在本探索runner中缓存不变的原始code解码结果，已知有限的零替换合并赋值；新code仍逐个经原生SAE decode，并保留正式`edit_features`的两步BF16舍入。每题探针必须与正式接口补丁逐值相等。不同执行版本按源码SHA保存在`results/code/`。

## 重跑与续跑

从仓库根目录执行：

```bash
python -m scripts.SAR.prepare_plan
python -m scripts.SAR.exhaustive
python -m scripts.SAR.build_exhaustive_report
python -m scripts.SAR.verify_results
```

`exhaustive`按每题逐组合续跑；`--limit 1`可先核验首题。已完成题目不会重复测量，每完成一题自动更新报告。报告入口也允许单独生成部分完成的报告，并明确标注完成题数。`verify_results`要求50题全部完成。后台运行使用tmux会话 `sar-exhaustive-20261115`，原intervene/generate矩阵会话独立保留。

本次计算期间single口径发生确认，独立运行`python -m scripts.SAR.watch_reports`以实时刷新确认后的报告；tmux会话为`sar-reports-20261115`，日志为`results/report_builder.log`。该脚本在50题完成、推理进程退出后写入最终报告并执行完整核对。

## 历史脚本

`legacy/`保留此前五数据集批次、中文报告、四格模式搜索报告及原始SAR算法的脚本快照；`pair_search.py`为已有测量搜索入口快照。`script_sources.json`记录原路径。历史脚本保留原路径语义，仅用于追溯；本次运行入口是上述三个模块，结果统一写入本目录的`results/`。

正式Capture实现仍在`tasks/capture/`，模型评分与特征编辑复用`saebase/api/`；这里保存用户指定的探索脚本。
