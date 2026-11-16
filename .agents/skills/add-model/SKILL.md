---
name: add-model
description: 接入或修改 SAEBase 模型构造、配置及供数据集和任务使用的输出接口时使用。
---

# 接入 Model

## 目的和主要功能

将模型接入现有构建和注册，使目标科研任务取得所需输出。

## 适用范围

用于模型接入或修改。输入为模型与权重来源、目标数据集和任务、已有相近实现；工作模式、检查范围和文件处理采用根 AGENTS.md。

1. 阅读 `saebase/registry.py` 的 `ModelPlugin`、`models/build.py`、`models/plugins.py` 和相近实现，沿用 `construct(cfg)` 与 `assert_compatible(cfg)` 接口。
2. 制定接入方案时，先明确目标任务如何选择、加载和使用模型与 SAE，据此确定输入、返回能力、权重身份和需要检查的条件；缺少条件时提供所选资源的具体原因和处理建议。再在现有模型目录内实现，沿用模型兼容性接口。
3. 在所属系列声明模型，现有系列由 `models/plugins.py` 统一登记；模型预设放 `conf/model/catalog.yaml`；按实际变化更新 `conf/models_log.md` 摘要。
4. 围绕目标任务的正常使用选择最小检查。合成输入足以确认接口时直接使用；需要真实权重的部分依据相应授权执行。
5. 修改已有模型时检查实际受影响的调用，复用公共接口已有保证。
