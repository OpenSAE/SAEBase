---
name: add-dataset
description: 接入或修改 SAEBase 数据集实现、样本结构、数据划分及任务使用接口时使用。
---

# 接入 Dataset

## 目的和主要功能

将数据集接入现有注册和配置，使实际科研任务取得所需样本。

## 适用范围

用于数据集接入或修改。输入为数据来源、目标任务及已有相近实现；工作模式、检查范围和文件处理采用根 AGENTS.md。

1. 阅读 `datasets/datamodule.py`、`datasets/plugins.py`、`saebase/registry.py`、相近实现和目标 Task 的样本接口。
2. 制定接入方案时，先明确目标任务如何实际读取和使用样本，据此确定样本结构、标签、划分、来源与准备结果。需要保存的资料同时确定读取方法；输入不足时指出具体数据和准备动作。再在 `datasets/<DatasetName>/` 实现，沿用现有 DataModule 和任务准入接口。
3. 在 `datasets/plugins.py` 注册数据集；需要别名时使用已有别名注册接口。
4. 配置放 `conf/dataset/catalog.yaml` 的对应预设，数据路径使用 `paths.data_dir`；按实际变化更新 `conf/datasets_log.md` 摘要。
5. 选择目标 Task 的代表性输入检查实际消费结果，复用注册和准入接口已有保证；在数据说明和配置中记录实际来源与划分假设。
