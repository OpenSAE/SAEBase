---
name: add-dataset
description: 当需要新增、接入或更换数据集及其 DataModule、split、label schema 和 dataset-task 准入时使用。
---

# 接入 Dataset

1. 阅读 datasets/datamodule.py、datasets/plugins.py、saebase/runtime/registry.py、已有最相近的数据集和实际 Task 消费的数据接口。
2. 先写清样本结构、dtype、label mapping、split 来源，以及哪些信息属于 Dataset 而非 Task。
3. 在 datasets/<DatasetName>/ 添加实现、数据准备入口及 dataset_task_policy.py，保留现有 DataModule 插件契约。
4. 在 datasets/plugins.py 注册 register_dataset；需要 alias 再调用 register_dataset_alias，不修改 saebase.registry 或 datamodule 内的具体数据集分支。
5. 添加 conf/dataset/<name>.yaml，data_path 由 paths.data_dir 决定，数据文件不进入仓库；按需同步 conf/datasets_log.md 人类摘要。
6. 先验证注册表识别，再用小的合成/真实样本验证 train/val/test 的结构、label 与边界条件；对评测集重用行为明确说明。
7. 运行 python .cursor/hooks/validate_repo.py --full；条件满足时运行具体 Task 的 pipeline checks，不得绕过兼容性门控。
8. 汇报路径、标签和 split 假设、是否需要真实数据、验证命令及尚未验证的运行时行为。
