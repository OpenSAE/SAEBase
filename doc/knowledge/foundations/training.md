# 训练 · 损失、优化器与调度

当前分类任务使用 `tasks/vision/classification/classification_module.py` 的 LightningModule，通过 `tasks/vision/classification/run.py` 执行训练。

默认 SGD 学习率为 `0.025`，动量为 `0.9`，权重衰减为 `5e-4`；余弦退火 `t_max=128`，总训练轮数 `128`，标签平滑 `0.1`。当前 CUB 数据批大小为 `64`。这些数值对应当前仓库配置。

分类模型产生 logits `z`，Softmax 转换为概率 `pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)`。标准单标签交叉熵为 `L = −log(pᵧ)`。实际损失通过 `F.cross_entropy(logits, y, label_smoothing=self._label_smoothing)` 计算。

关键代码：`ClassificationModule.training_step`、`ClassificationModule.configure_optimizers`。参数与返回语义可在 Code Tree 节点内部展开阅读。
