# 复现 · 配置、随机数与实验产物

训练入口为 `python -m dlbase.train`。Hydra 组合 `conf/` 中的配置；默认随机种子为 `42`。各次运行记录模型、数据集、任务路径、配置哈希与执行环境。

`dlbase/experiment_paths.py` 创建类型化产物目录：`checkpoints/`、`metrics/`、`tables/`、`figures/`、`logs/`、`metadata/`、`reports/` 及 `tmp/`。`metadata/run.json` 保存运行元数据，`reports/summary.log` 汇总人类可读的实验结果。

`harness/environment/probe.py` 采集机器实际资源，便于比较跨设备的执行条件。
