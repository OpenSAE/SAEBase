# 工程维护与批次入口

## 目的和主要功能

说明软件发布、分发、文档、代码检查与具体批次启动的使用方式。

## 适用范围

用于软件工程维护与本页列出的批次启动；科研方法和科研结果判断由所属 Task 负责。

工程操作使用四个入口，共用各自的读取、处理和报告程序：

| 文件 | 主要操作 |
| --- | --- |
| `release.py` | 软件版本、来源、发布条件与资产登记 |
| `distribution.py` | `contents` 检查包内容，`install` 检查仓库外安装，`roundtrip` 从源码包重建并检查 |
| `documentation.py` | `--check` 检查文档链接；`support --check` 或 `support --write` 根据评测证据检查或更新支持说明 |
| `quality.py` | 选定代码的格式和静态检查；`select --base <提交> --head <提交>` 选择持续集成检查 |

科研参数转换归所属任务命令，数据准备使用 `saebase-prepare-prompts`。工程开发的完整用法见[贡献指南](../.github/CONTRIBUTING.md)。下文介绍已有批次启动。

普通矩阵、whiten 和 SAR 队列共用 `saebase/runtime/environment.py` 启动模型子进程。
本机解释器和关键依赖版本统一配置在 `conf/runtime/environments.yaml`；默认使用
`/venv/main/bin/python`，Qwen3.5 使用 `/workspace/envs/saebase/qwen35/bin/python`。
Runtime 在加载权重前检查依赖版本及所需模型架构，检查不通过时保留明确的失败记录。
直接使用矩阵的 `--model` 参数时也会选择对应环境。

主进程保持原有环境，子进程结束后继续使用默认环境运行其他模型。正常结束、报错、
超时和收到停止信号时均清理本次创建的进程组。每个模型日志旁保存
`run-<model>.environment.json`，包含解释器、依赖版本和实际包位置；每条实验的
`metadata/manifest.json` 同时记录实际运行环境。环境本体与依赖清单保存在
`/workspace/envs/saebase/`，实验目录保存测量和运行记录。

新的 capture 批次使用正式能力：

```bash
python -m tasks.capture.matrix --config conf/experiment/capture_matrix.yaml --capture-mode whiten --tag my_whiten_batch
```

批次配置只列本次模型/SAE、数据集选择和执行预算。每个组合仍通过 SAEBase 的 `conf/config.yaml` 与公共 Runtime 运行，自动先 precheck，再 capture。批次复用每个模型的加载会话，并保留失败组合和可继续执行的记录。

批次调度由 `tasks.capture.matrix` 维护。配置选择模型、SAE、数据集和预算，各组合通过正式 Runtime 与 Precheck 执行。


## GPU 空闲后运行 Capture 矩阵

`capture_queue.py` 调用正式的 `tasks.capture.matrix`，读取指定矩阵配置中的 Dataset 和 Model/SAE/层位选择。每个模型启动前等待外部计算进程全部离开 GPU，显存占用不超过 1024 MiB、利用率不超过 5%，并连续保持至少 300 秒（5 分钟）；命令行不接受低于 300 秒的门槛；`--wait-pid` 还可等待指定在用批次退出，避免模型切换间隙抢占显卡。探测失败时继续等待。

```bash
python -u scripts/capture_queue.py \
  --sar-source /absolute/path/to/SAR/sar.py \
  --report-dir /absolute/path/to/matrix-report \
  --tag sar_answer_matrix \
  --readout answer_mean_logprob
```

默认选 16 个特征、抽 384 个子集，其中 256 个拟合、128 个留出；Dataset 仍使用各自已有的固定样本选择。`answer_mean_logprob` 对标准答案全部 token 做 teacher forcing，计算平均条件 log probability。选择题使用正确选项正文，GSM8K 使用最终答案，HumanEval 使用原始标准代码补全；题目输入与特征选择不加入答案。没有标准回答的记录（包括当前 HarmBench、XSTest）标为 unsupported，不能充作成功测量。

运行产物仍保存到 `experiment/<dataset>/<model>/capture/<tag>_<seed>/`。报告目录保存 `effective_matrix.yaml`、`launch_history.jsonl`、`status.json`、各模型日志和逐组合完成记录。查看批次时先读取逐组合完成情况及原因，再使用对应运行的正式结果判断可复用内容；日志用于定位实际错误。维护调度与报告功能时，保证这些记录能够支持继续批次和找到所需产物。再次启动相同配置会读取已有完成记录；失败组合保留旧目录并创建新尝试。模型间串行执行，SAR 需要完整后续网络，关闭普通 capture 的尾层卸载。默认不设置单模型的总时限；可用 `--model-timeout-seconds` 设置正数，超时只终止本脚本创建的进程组并继续后续模型。

`--check-config` 只检查所有组合，不加载权重、不执行原版估计器或等待 GPU。用于 tmux 时将标准输出和错误重定向到相应 capture `_tmp/` 下；用 `tmux attach -t <会话名>` 查看进度。

`capture_queue.py --capture-mode sar` 使用原版 SAR 源码；`--capture-mode whiten` 和 `--capture-mode non_whiten` 使用对应捕获方式。组合数量来自配置，队列共用正式矩阵调度。


## 可运行演示

`scripts/examples/first_experiment.py` 使用合成模型完成评测、统计与报告，支持 `--dry-run` 和 `--output`。`scripts/examples/external_providers.py` 展示模型、SAE和指标的登记与实际消费。演示在源码环境运行，合成组件由 `tests/fixtures/synthetic.py` 提供，通用调用见[首次实验教程](../doc/code/development/first-experiment.md)。
