---
name: add-model
description: 当需要新增、替换或扩展模型及其构造、配置、兼容矩阵和输出接口时使用。
---

# 接入 Model

1. 阅读 saebase/registry.py 的 ModelPlugin、models/build.py、models/plugins.py 和相近模型实现，确定 construct(cfg) 与 assert_compatible(cfg) 的实际契约。
2. 明确输入 shape、dtype、输出语义（logits/embedding/probability）、预训练权重来源及目标 Dataset/Task。
3. 在 models/<name>/ 实现模型和 model_task_dataset_compatibility.py；通用模型构造层不引入名字特判。
4. 在 models/plugins.py 注册 register_model；添加 conf/model/<name>.yaml；按需更新 conf/models_log.md 人类摘要。
5. 用合成输入进行 CPU、小 batch 前向与形状、数值有限性、输出语义检查；依赖额外权重时写明无法离线验证的部分。
6. 对目标 Dataset × Task 组合运行真实的准入检查；运行 python .cursor/hooks/validate_repo.py --full。
7. 若原模型支持其他任务，补充最小回归；未运行完整训练时明确标注。
