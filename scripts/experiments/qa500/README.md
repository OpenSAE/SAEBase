# QA500 标准问答评测实验

QA500 是一个 **具体实验执行脚本集合**，不是 Dataset、Model、Task、Runtime 或 Harness。
本目录只拥有科研组件的选择、运行矩阵、Frozen cohort、结果汇总、
跨 Run 审计和 HF staging。科学实现调用现有的 SAEBase 组件。

## 目录职责

| 层次 | 权威来源 |
| --- | --- |
| 独立 Dataset 和上游字段适配 | `datasets/<Dataset>/qa_adapter.py` |
| Dataset 身份、Prompt 记录协议及通用读取 | `saebase.registry`、`saebase/data/qa_records.py`、`qa_source.py` |
| Model tokenizer 和 Chat template | `models/qa_input.py` |
| Capture 的真实 SAE 特征观测 | `tasks/capture/qa_capture.py` 和 `saebase.api` |
| QA 评分与 Core 统计 | `tasks/benchmark/qa_scoring.py`、`qa_metrics.py` |
| Neuronpedia 解释来源 API | `saebase/integrations/neuronpedia.py` |
| 环境检测、隔离与 Runtime 产物布局 | `saebase/runtime/hardware.py`、`supervisor.py`、`experiment_paths.py` |
| QA500 所有科学配置 | `conf/experiment/qa500/` |
| 入口 Bash 脚本 | `scripts/experiments/qa500.sh` |
| 治理、独立验证、授权及验收 | `harness/github/work_plan/22_qa500_standard_evaluation_pipeline.md` |

实验的通用资源能力必须保持对 QA500 零依赖。
其他实验可以复用 Dataset、Model、Task 和 Runtime，而无需导入此目录。

## 全新开发环境启动

支持 Python 3.11/3.12 的 Linux 系统。本脚本创建虚拟环境并安装 Python 依赖。
运行 GPU 任务需要现成的可用 NVIDIA 驱动及支持该设备的 PyTorch CUDA 构建，
不能自动获得 root 权限安装内核驱动。Ne­uronpedia、Llama 和 GPQA 的受限
权重及数据需要根据实际上游许可单独授权。

```bash
# 单次显式获取分支，适用于 origin 默认仅跟踪 main 的仓库
git fetch origin refs/heads/feat/qa-evaluation-500-v1:refs/remotes/origin/feat/qa-evaluation-500-v1
if git show-ref --quiet --verify refs/heads/feat/qa-evaluation-500-v1; then
  git switch feat/qa-evaluation-500-v1
else
  git switch --track -c feat/qa-evaluation-500-v1 origin/feat/qa-evaluation-500-v1
fi
git pull --ff-only origin feat/qa-evaluation-500-v1
git rev-parse HEAD
bash scripts/experiments/qa500.sh preflight
# 完整 GPU 执行需要单独明确授权
# bash scripts/experiments/qa500.sh full --allow-local-gpu
```

`preflight` 负责本地导入、离线负例、Harness Contract、机器可用资源与真实 CUDA
kernel smoke；它不会下载模型。主命令按顺序执行 Prepare、10题 Pilot、
完整 QA Baseline、Capture、Benchmark、Intervene、Generate、Explain、Audit 与
仅本地 HF Stage。正式 HF Publish 是单独批准的动作。

**资源和可移植性边界**

- 每个模型采用独立操作系统子进程，避免上一个 CUDA 状态污染下一个模型。
- CUDA OOM 只允许固定预算的原模型、原 SAE、原精度新进程重试；已有合法
  sample 结果通过原始哈希恢复。不能通过换模型、减少 500 题、裁剪输入或换精度
  达成假完成。
- 若 GPU 内存不够，且同精度 Offload 尚未对当前 Hook/Backend 验证，则明确
  标记 `blocked_resource`；不能暗示所有 16GB 或 24GB GPU 必然可执行全矩阵。
- Machine profile 记录真实 GPU 名称、计算能力、显存空闲量、PyTorch/CUDA
  版本、CPU RAM 与磁盘余量。未知值保持 null。
- 当项目 `assets.lock.json` 尚未固定所有上游 commit 时，首次下载结果
  **只具备本地 manifest 内的修订身份**，不具备跨机器正式锁定的证据。
  公共 HF 发布必须拒绝这种状态。
- 模型需要的 CPU Offload、不同精度加载与 Native Tracer 环境仍需真实
  硬件兼容性验收。代码准备成功不代表 20 位点实跑成功。

## 冻结协议

全部数据集来自既有 Dataset Registry 中可复用的 QA Adapter。
`conf/experiment/qa500/datasets.json` 选择十个来源，每源目标 500 个唯一真实题目。
`sites.json` 选择十模型和二十个 SAE/Layer/Hook 位点；`tasks.json`
固定任务配额及方法；`resources.json` 限定允许的恢复策略；
`assets.lock.json` 记录固定 checkpoint 与数据来源的发布准备程度。

默认规模为 50,000 个模型×问题 Baseline、100,000 个 SAE 位点×问题 Capture、
200 份 Core benchmark 统计、4,000 次 Intervene、1,000 次 Generate。
Explain 仅报告由 Neuronpedia 精确匹配来源提供的 annotation，
不把第三方描述当作独立的因果解释验证。
Circuits 和 Train 是额外授权的原生工作器，运行入口如下。

```bash
bash scripts/experiments/qa500.sh native --allow-native-compute
```

## 正式 Runtime 产物

单次真实 Task Run 通过 `saebase.runtime.experiment_paths.run_leaf` 获取路径。

```text
experiment/
  <dataset>/
    <model>/
      <task>/
        qa500/<sae_key>/L<layer>/qa500_20261031/
          metadata/run.json
          metadata/*_config.json
          metrics/*.jsonl
          logs/*_failures.jsonl
          reports/*_summary.json
  _studies/qa_standard_500_v1/
    inputs/<dataset>/questions.jsonl
    inputs/<dataset>/manifest.json
    metadata/receipts/
    logs/
    staging/
```

研究输入和跨 Run index 属于 study 级元信息；Feature 捕获、指标、干预
和 Native 真实结果均属于已有 Task 的 Run leaf，使用该 Run 的 typed artifacts。
禁止回退到旧的 `experiment/qa500/results/` 平行结果布局。

模型不兼容、HF gated 授权缺失、输入长度失败、SAE Hook 不匹配及 OOM
均应保留原始运行证据。每一个执行状态必须区分
`implemented`、`self-tested`、`independently-verified`、
`blocked`、`published`。未执行的组合不计为通过。

## 独立验收与发布

```bash
python -m pytest -q tests/workplan/wp22 harness/github/work_test/test_wp22_architecture.py
python -m harness.contracts.registry check \
  --plan harness/github/work_plan/22_qa500_standard_evaluation_pipeline.md
python -m scripts.experiments.qa500.report audit
python -m scripts.experiments.qa500.report stage
```

实际发布只可以在独立审计完成、项目完整 asset lock 和人工批准后使用

```bash
python -m scripts.experiments.qa500.publish \
  --stage experiment/_studies/qa_standard_500_v1/staging --approved
```

历史 SAEBase-Evaluation 的既有配置保持不变，所有重试和冻结输入的证据不得静默覆盖。


> QA500 研究编排位于 scripts/，不属于可安装的 SAEBase Python 公共 API。测试集中于 `tests/workplan/wp22/`。运行环境与生成元数据均为非源码工件。
