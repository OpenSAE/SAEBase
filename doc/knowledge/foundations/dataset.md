# 数据 · CUB-200-2011

**研究内容：** 从真实鸟类图像与官方标签构建 200 类细粒度分类任务。

当前 `conf/dataset/cub.yaml` 设置 `num_classes=200`、`image_size=448`、`batch_size=64`。图像与标注读取位于 `datasets/vision/CUB_200/CUB_200.py`，数据加载与变换位于 `datasets/vision/CUB_200/datamodule_setup.py`。

官方 Train 用于训练数据，官方 Test 用于测试数据。默认 `train_val_ratio: null`、`val_split_fraction: 0`，Lightning validation 当前采用官方测试划分。独立验证集可由官方训练集再划分，配置项支持 `val_split_fraction` 或 `train_val_ratio`。

训练图像采用训练阶段变换，评测图像采用确定性预处理。对应概念见 `term.cub200`、`term.official_split`、`term.validation_set`、`term.augmentation`。
