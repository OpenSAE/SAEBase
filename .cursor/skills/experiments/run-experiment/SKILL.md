---
name: run-experiment
description: 当用户要求直接运行训练、评估、sweep、长耗时科研任务或查看实际实验结果时使用。
---

# 执行研究实验

1. 阅读目标 conf YAML、Task 运行入口、数据路径和兼容性检查。先确定运行的 Dataset × Model × Task 是否可用。
2. 使用当前训练 Python 检查解释器及依赖。先运行 python -m dlbase.train --cfg job 查看有效配置；必要时先运行最小 smoke。
3. 确认 paths.data_dir 对应真实数据；不要在项目内复制海量数据。检查实验叶子 experiment/<dataset>/<model>/<task.path 层级>/<sub_name>_<seed> 是否已存在。
4. 单次实验保留 experiment.sub_name=basic，sweep 用短且唯一的变化参数后缀。已有产物的叶子默认由 dlbase.experiment_paths 拒绝覆盖；必须经用户明确允许才设置 experiment.overwrite=true。
5. 用户要求实际运行时由 Agent 执行 python -m dlbase.train；默认前台。后台仅在明确要求时使用 tmux，stdout/stderr 先写到对应 task 层级的 _tmp/<session>/（不可写入全局 experiment/tmp/）。完成后检查 metadata/run.json、reports/summary.log 和 checkpoints/。
6. 分别报告运行配置、训练是否真正完成、best checkpoint、summary.log 和观察到的评测指标。不要把只有配置检查或短 smoke 的结果说成最终训练结果。
7. 长作业保留命令、环境、日志、session/checkpoint 与恢复路径；若当前执行环境无法持续等待，明确说明状态，不承诺后台自动继续监控。
