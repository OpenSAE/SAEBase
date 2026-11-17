# Capture筛选、观测数量与答案评分

## 目的和适用范围

说明正式Capture的候选筛选、输出数量和四种末token模式。运行入口为`python -m saebase.runtime.train task=capture`，配置位于`conf/task/capture.yaml`，结果由`CaptureTask.read_samples`和`export_report`读取。

## 参数

- `task.filter=10`：默认应用已有SAR语义／原生密度及参考题高频过滤，再按对应模式选择10个候选。
- `task.filter=all`：跳过上述过滤，使用全部SAE特征。Whiten仍排除std=0的特征。
- `task.capture_count=null`：默认输出数量跟随filter；filter=all时对应全部可用结果。可以单独设为1、5等数字；单特征模式的数字不得超过数字filter的候选数。SAR输出的是pair数，可以超过候选特征数，最多为候选间不同pair的数量。
- `task.top_feature_num`：保留旧的输出数量别名；显式capture_count优先。新配置推荐使用capture_count。
- `task.sar.cohort_filter.reference_sample_count=100`：筛选参考题数量，独立于候选／输出特征数量。
- `dataset.sample_count`：实际观测题目数量，保持Dataset原有配置。

语义／密度规则复用`tasks/capture/sar_filter.py`；mean/std使用参考题全部prompt token，频率规则使用参考题末token的有符号z值，默认z>1的题目比例达到50%时排除。各项过滤细节仍可在task.sar.feature_filter、cohort_filter中设置。std=0排除，不设正数std下限。

## Capture模式

- `mode=whiten`：在过滤允许的特征中，按|zscore|排序。
- `mode=non_whiten`：在过滤允许的特征中，按|native activation|排序。
- `mode=g_s`：候选默认按|zscore|取得，逐个清零一个候选，其余所有特征保持原值；按删除后标准答案的平均token log概率g降序输出。delta_g为删除后g减原始全保留g。
- `mode=I`：穷举候选全部保留mask，对每个pair遍历其他候选的所有固定背景，计算I=g11−g10−g01+g00，每对取最大I后按原值降序输出。10个候选对应1024个状态、45对×256背景。mask、g、背景和来源保存到NPZ及结果JSON。

g(S)和I模式默认对标准答案进行teacher forcing，不加EOS；改变的是原prompt最后一个token在选定SAE位置的激活，保留原重构残差。`answer_backend=full_context_tail`缓存完整序列的下层激活，使用相同batch/序列形状计算尾层，并与原生完整前向核对；不支持该尾层接口的后端使用完整前向。

`mode=sar`使用原SAR抽样回归方式，需要显式可信的sar.py；其pair系数与穷举四格I是两种指标。穷举候选数默认上限16，过大候选集合会在测量前明确报错，不能用回归预测冒充全遍历。`filter=all`用于大型SAE的穷举SAR时同样受此上限约束。

未包含新filter字段的历史调用保留旧的候选／输出行为。下游干预／生成调用显式传入filter、capture_count时会转交Capture；没有这些字段的历史任务继续沿用已有选择，避免改变已启动批次。`output_scope=all_tokens`属于原生稀疏轨迹接口，使用filter=all及其max_active_per_token参数。

## 使用

```bash
# 默认筛选10个，输出10个whiten特征
python -m saebase.runtime.train task=capture task.execute=true
# 同一候选池，仅输出g(S)最高的一项
python -m saebase.runtime.train task=capture task.execute=true task.mode=g_s task.filter=10 task.capture_count=1
# 不过滤，输出原始activation第一名
python -m saebase.runtime.train task=capture task.execute=true task.mode=non_whiten task.filter=all task.capture_count=1
```

模型和数据集仍通过标准model、dataset配置选择。要复现此前五个QA数据集的固定题目和Llama L21配置，使用`scripts/SAR/run_capture.py`；已有50题的离线复核使用`scripts/SAR/check_capture_reproduction.py`。

批量运行 `python -m tasks.capture.matrix` 时，矩阵配置的 `matrix.dataset_overrides` 合并到每个数据集预设。例如 `dataset_overrides: {sample_count: 100, sampling_seed: 42}` 将本批各格设为100题，不改变数据集公共默认值。模式通过 `--capture-mode` 指定。

正式 Capture 默认开启 `task.performance.enabled=true`。Precheck 先做对应模式的真实性能探针，执行器按精度约束选择尾层缓存与状态 batch；`performance.enabled=false` 保留原标量方式。每题耗时与共享标准化、筛选、解释、前向、排名、写入等阶段分别记录。I 去重后仍保存完整1024状态响应；额外记录真正执行的独立 code 状态数。具体预算、报告及数值阈值见 `tasks/precheck/README.md` 的“性能预检与自动调优”。

## RSS：固定目标与新增上下文

设置 `task.mode=rss`、`task.filter=all`、`task.output_scope=last_token`，提供 `task.prompt`、`task.rss.contexts` 和可信本地原版 `task.rss.source`。默认 RSS 参数为 `M=400, K=20, rho=kappa=0.8, tau=1e-5`。每个 variant 为 `context + 空格 + clean prompt`，在目标最后一个 token 上读取完整原始 SAE codes。

`rss_case.md` 提供 RSS 全部选集及数值上集中于各 variant 的解释候选；`rss_case.json` 保存原始解释、recurrence、方向、排除原因；`rss_activations.npz` 保存完整 clean、variants 和 delta。展示候选要求 recurrence < rho、正变化 > tau、相对其他变体至少五倍，并按变化幅度排序，保留所有符合数值条件的特征。语义是否对应具体上下文由阅读解释判断。

本次五个鲜明上下文的临时完整入口：`python scripts/RSS/run_contrast.py`。该脚本使用 RSS 公开 demo 的 GPT-2 加载设置，通过 `CaptureTask.run` 生成每一个案例，输出到新的 `experiment/RSS/` 目录。
