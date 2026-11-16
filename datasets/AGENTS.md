# 数据集工作说明

## 目的和主要功能

说明数据集实现、接入和产物的职责。

## 适用范围

适用于 `datasets/` 的阅读与开发；工作模式沿用根 AGENTS.md。

- `datasets/<name>/` 保存数据生成、准备、读取、Token 编码和划分，`datasets/plugins.py` 接入统一 Registry。
- 源码通过 `saebase.datasets` 命名空间导入；`saebase/datasets/` 维护兼容导入，保留第三方 Hugging Face `datasets` 包的正常使用。
- 配置放 `conf/dataset/catalog.yaml` 的对应预设。自然数据接入在所属目录的 `__init__.py` 声明准备、选择及支持任务，复用公共来源实现；复杂准备逻辑保存在该数据集内。数据产物记录实际来源、模型身份、版本及划分；Prompt 数据按已有 manifest 和 `prompts.jsonl` 结构组织。
- 制定准备、读取或数据检查功能时，从目标 Task 实际取用样本的过程确定输出：样本、划分与来源能够被任务读取；准备失败时能定位具体数据源、输入位置及需要完成的准备动作。保存的数据说明与准备结果沿用现有样本读取入口，检查时用目标 Task 实际消费。
- 接入新数据集时使用 `add-dataset` 技能。
