# 模型 · ResNet-18 分类器

当前模型实现位于 `models/vision/resnet18_mlp/resnet18_mlp.py`，CUB 模型配置位于 `conf/model/resnet18_mlp.yaml`。当前默认 `head_style: linear`，输出类别数由 `dataset.num_classes` 设置为 200。

`Resnet18MLP` 持有 `self.resnet18` 主干、`self.head_mode` 分类模式、`self.mlp` 备选多层分类头与 `self.backbone_drop` 特征层变换。`forward(x, epoch_id, **kwargs)` 接收图像张量；线性模式经 ResNet 各层和平均池化后输出 logits，多层模式将 ResNet 特征传给 `MLPModuleResnet`。

残差模块的常用概念式为 **y = F(x; W) + x**。主分支完成参数化变换，捷径将输入送至求和点；尺寸改变时可以采用投影捷径。

Context：[ResNet-18](https://atheavenma.github.io/opensae/openresearch/atlas/context.html?focus=term.resnet18)，对应代码：[ResNet 前向方法](https://atheavenma.github.io/opensae/openresearch/atlas/code.html?focus=code%3Amethod%3Amodels.vision.resnet18_mlp.resnet18_mlp%3AResnet18MLP.forward)。
