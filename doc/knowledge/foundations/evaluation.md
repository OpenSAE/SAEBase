# 评测 · 分类结果与数据独立性

训练损失记录优化轨迹。验证指标支持模型选择与超参数配置。最终测试指标描述确定方案后的分类表现。主要指标为 Top-1 accuracy，Top-5 提供候选覆盖比例。

当前默认 `val_split_fraction: 0`，Lightning validation 使用官方 Test 数据。研究者可以从官方 Train 建立独立 Validation，完成模型选择后在官方 Test 记录最终结果。

模型性能验收需要明确准确率目标、比较基线、运行预算、随机种子与评测范围。相关接口位于 `tasks/classification/post_train_eval.py` 和 `dlbase/summary_log.py`。
