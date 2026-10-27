# 实验输出层级与可复现性协议

这是可选的开发管理工具；**真正运行时的通用协议在 `dlbase/experiment_paths.py` 和 `conf/`**，不会让训练去 import `harness/`。

## 目录形式

```text
experiment/
└── CUB_200/
    └── resnet18_mlp/
        └── classification/
            ├── _tmp/
            │   └── <session>/            # tmux 启动前日志，task 级别
            ├── basic_42/
            │   ├── .hydra/               # Hydra 自带配置记录
            │   ├── checkpoints/          # best.ckpt、epoch*.ckpt
            │   ├── metrics/              # json / jsonl
            │   ├── tables/               # csv / parquet / tsv
            │   ├── figures/              # png / pdf / svg
            │   ├── logs/                 # TensorBoard 和普通 logs
            │   ├── metadata/             # run.json，配置哈希
            │   ├── reports/              # summary.log、说明
            │   └── tmp/                  # 当前 run 的临时结果
            └── experiment_group/
                └── basic_42/             # task.path=classification/experiment_group
```

可通过 `task.path: classification/subtask/...` 深入任务层级。不得通过 `../`、绝对路径或不合法名字越界。
`experiment/tmp/` 全局目录已废弃；不会自动搬运历史文件，以避免破坏旧实验。

## 产物分类与说明

每个运行叶子自动生成 `metadata/run.json`，记录 Dataset、Model、Task、分层、seed、配置 hash、目录说明和各产物类别。`reports/summary.log` 是人类可读的运行摘要。若新增逻辑子文件夹，必须有固定描述并在 `metadata/run.json` 或对应 manifest 中说明；不得通过猜测内容臆造研究 Context。

`metadata/run.json` 的 note 必须参考 `doc/knowledge/context_graph.json` 的真实目标，或标明“运行元信息”；不要把目录描述当科研结论。

## 覆盖策略

相同 `dataset/model/task.path/sub_name/seed` 的叶子可能包含旧 checkpoint。**默认拒绝覆写**；确实需要覆盖时由用户明确设置 `experiment.overwrite=true`。不得因 Smoke 误删旧实验。
若仍有旧脚本使用根目录 `summary.log`、`best.ckpt` 或 `tensorboard/`，应先使用当前项目内置调用链升级，不把本模块当成独立训练入口。

## 临时日志

需要后台时，先由任务配置推导 task root，再建 `_tmp/<session>/` 记录 stdout/stderr。任务启动前先创建这个目录；不要将 stdout 直接指向会被下一次运行清理的叶子 `tmp/`。

**全部 Task 的标准目录与元信息都由 `dlbase.train` 调用 `prepare_run` 初始化**，各 Task 不需要再自行调用清理或目录构造；生产代码只使用 `artifact_path` 保存对应产物。

## 按功能继续细分

需要在某个 run 内建立新的分析上下文时，可以调用公共 `dlbase.experiment_paths.create_analysis_group`：

```python
from dlbase.experiment_paths import create_analysis_group
group = create_analysis_group(cfg, cwd, "feature_study/ablation",
    context_ids=["g.cub.optimizer"],
    description="目标：以受控消融比较分类模型配置，并记录实验依据。")
```

这会创建 `analysis/feature_study/ablation/`，再次包含 checkpoints、metrics、tables、figures、logs、metadata、reports、tmp，并自动写入 `metadata/context.json` 和更新运行 manifest 的 `analysis_groups` 索引。重复调用不得静默改变已有 Context。

`dlbase` 只验证 ID 格式与描述形式，不读取研究文档；使用 `harness/experiments/validate.py` 对照 Knowledge tree 实际节点验证引用。这样不会让训练流程去依赖 `doc/` 中的动态说明文件。
