# WP22 — QA500 标准实验（科研组件与实验组合解耦）

结构权威：`scripts/experiments/qa500/` 仅保留特定实验的执行编排；`datasets/`、`models/`、`tasks/` 和 `saebase/` 提供可重用能力。`conf/experiment/qa500/` 是唯一研究配置源，运行入口归 `scripts/`。Runtime 正式产物需遵循既有 `saebase/runtime/experiment_paths.py`。

## 结论与操作边界

本 Work Plan 是 SAEBase v0.6.5 之上的固定 QA500 新 cohort。目标是 **10 个真实 Hugging Face QA 数据集 × 每集 500 题 × 10 个基础模型 × 每模型 2 个真实 SAE 层**，即每轮最多 5,000 个标准输入、50,000 个模型回答与 100,000 个 Model/SAE/Layer × 问题 Capture 测量；分别执行固定小样本 Intervene (4,000)、Generate (1,000)、Explain (每 site × dataset 2 个)，以及两个独立原生任务试验。不得把历史 Stage D 48-site、Research Lab 500-prompt 和 QA500 视为同一输入群体。

核心政策是**本地 agent 只执行**。研究人员和本计划已固定了数据、模型、层、采样种子、参数、任务预算、输出结构、失败策略与发布门槛。Agent 不自行选择某个新数据集、替换不可用模型、改题目或评分、降批量、调参数、挑看起来好看的 feature、重新诠释 L0/MSE、生成解释或绕过权限。遇到不兼容、gated 权重、超预算、错误/缺失的 HF 字段，应生成固定失败收据并停止该执行单元。

本阶段 **发布代码，不等于真实 GPU 已验收，不等于七种任务在所有位点都有科学结果，不等于 HF 发布完成**。所有相关报告需要区分 implemented/self-tested/independently-verified/accepted/published。

## 固定协议与唯一权威

| 组件 | 已确定的来源 |
| --- | --- |
| 统一协议 | `conf/experiment/qa500/protocol.json`，`scripts/experiments/qa500/runner.py plan`，二者由离线验收检查一致性 |
| 数据下载和冻结 | `scripts/experiments/qa500/cohort.py`，十源均使用确切 upstream commit、固定 seed=20261031、分层抽样 500 |
| 模型层位点 | `scripts/experiments/qa500/capture.py SITES`，20 个明确模型/SAE/hook/layer |
| 模型输入与评分 | `models/qa_input.py + tasks/benchmark/qa_scoring.py`；正确的模型 tokenizer、Gemma IT chat template、候选 token 序列 log prob、贪心回答及类型化 scorer |
| 模型共享 Baseline | `scripts/experiments/qa500/measure.py`，每模型每数据集只评测一次，两个 SAE 层共享 |
| Capture/核心指标 | `scripts/experiments/qa500/capture.py`，使用真实 `saebase.api.load_registered`、`capture_features`、真实 SAE state SHA256、last prompt token |
| Intervene / Generate | `scripts/experiments/qa500/effects.py`，真实 `evaluate_edit` 与 `compare_steered_generation` |
| Explain | `scripts/experiments/qa500/annotations.py`，100 discovery/400 held-out，Neuronpedia 只作为外部来源文本，明确没有通过语义验收 |
| Neuronpedia | `saebase/integrations/neuronpedia.py` 和 `scripts/experiments/qa500/annotations.py`，必须精确匹配源 SAE 模型/source ID/权重 SHA256 |
| Circuits / Train | `scripts/experiments/qa500/native.py`，固定独立原生工作器与输入/训练资源；须另行授权、可能被环境阻塞 |
| Audit / HF stage | `scripts/experiments/qa500/report.py`，独立计数、源哈希、feature 范围、输入身份和聚合验证 |
| HF publish | `scripts/experiments/qa500/publish.py`，仅在用户明确批准及完整验证后追加新 cohort，现有 25 configs 不动 |
| 一键执行脚本 | `scripts/experiments/qa500.sh`；详见 `scripts/experiments/qa500/README.md` |
| 离线测试 | `tests/workplan/wp22/test_cohort.py`，`test_qa500_execution.py`，`test_qa500_protocol.py` |

任何代码与协议不一致，优先拒绝并提交差异报告，而不是让 Agent 选一个解释。

## Harness 通用 Contract 绑定

本计划仅选择 `harness/contracts/catalog.json` 中既有 Contract。不给 Agent 修改该 catalog、hidden evaluator、历史 WP18–WP21、Atlas Core/Knowledge 的权限。五个 Always-on 自动继承，每阶段增加相关条件 Contract。

```harness-contracts-json
{
  "registry": "saebase.harness.contract_registry.v1",
  "plan": [
    "CT-ERROR", "CT-EVIDENCE", "CT-PRIVACY", "CT-ENV",
    "CT-DETERMINISM", "CT-INTERFACE", "CT-DATA",
    "CT-CONFIG", "CT-RESOURCE", "CT-RELEASE",
    "CT-RECOVERY", "CT-NAMESPACE"
  ],
  "cases": {
    "Q01_PREPARE": {
      "tags": ["data", "determinism", "privacy"],
      "contracts": ["CT-DATA", "CT-DETERMINISM", "CT-PRIVACY"]
    },
    "Q02_PILOT": {
      "tags": ["environment", "resource", "validation", "interface"],
      "contracts": ["CT-ENV", "CT-RESOURCE", "CT-ERROR", "CT-INTERFACE"]
    },
    "Q03_CORE": {
      "tags": ["data", "config", "recovery", "evidence"],
      "contracts": ["CT-DATA", "CT-CONFIG", "CT-RECOVERY", "CT-EVIDENCE"]
    },
    "Q04_EFFECTS": {
      "tags": ["determinism", "interface", "validation", "evidence"],
      "contracts": ["CT-DETERMINISM", "CT-INTERFACE", "CT-ERROR", "CT-EVIDENCE"]
    },
    "Q05_NATIVE": {
      "tags": ["environment", "resource", "privacy"],
      "contracts": ["CT-ENV", "CT-RESOURCE", "CT-PRIVACY"]
    },
    "Q06_AUDIT": {
      "tags": ["evidence", "data", "validation", "recovery"],
      "contracts": ["CT-EVIDENCE", "CT-DATA", "CT-ERROR", "CT-RECOVERY"]
    },
    "Q07_PUBLISH": {
      "tags": ["release", "privacy", "resource"],
      "contracts": ["CT-RELEASE", "CT-PRIVACY", "CT-RESOURCE"]
    }
  }
}
```

验证命令

```bash
python -m harness.contracts.registry check --plan harness/github_work_plan/22_qa500_standard_evaluation_pipeline.md
python -m harness.contracts.registry resolve --plan harness/github_work_plan/22_qa500_standard_evaluation_pipeline.md --case Q03_CORE
```

## 运行矩阵及范围

| 阶段 | 固定输入 | 接受依据 |
| --- | --- | --- |
| Q01 Prepare | 10 个 500 题 HF 冻结集，完全相同的 500 ID 共享给全部模型 | 每源实际 500 个、唯一 ID，split/revision/hash 无变更 |
| Q02 Pilot | 全部 20 sites × 每数据集固定首 10 题 | 真实权重、真实前向、模型/SAE/hook/feature 对齐，记录 GPU 事实 |
| Q03 Core | 50,000 个 Baseline + 100,000 个 Capture | 全部 site/dataset 500 个记录，有限数值，准确原始 logits |
| Q04 Intervene | 每 site×dataset 20 题，自动按 SHA 顺序选，首个绝对值最大非零 feature 消融 | 实际 patch、identity replay、logit shift/KL；样本后选择须标注 |
| Q04 Generate | 每 site×dataset 5 题，同一 feature 按已冻结 ablate，greedy 16 tokens | baseline/steered 独立轨迹、温度 0、seed 固定，保存真实 token ID |
| Q04 Explain | 每 site×dataset 2 个 discovery 高频 feature，100/400 分割 | source label 和原始链接，如缺失则 explicit null，非解释正确性证明 |
| Q05 Circuits | Gemma 2 2B、BoolQ 固定 10 条短问题、原生 `circuit-tracer` | 原生 raw graph SHA，独立 transcoder 资源；失败按 blocked |
| Q05 Train | Gemma 2 2B L13、独立 Wikitext2 6400 语料、200 step TopK | 真实 native SAELens checkpoint，单独身份验证，非基准比较 |
| Q06 Audit | 所有真实结果与冻结输入 | 审计 `qa_report.py`；明细/聚合分开，失败/unsupported/partial 与成功分开 |
| Q07 Publish | 6 新 HF 数据配置，不包含第三方题干 | 人类另行批准、stage 完整、校验摘要、保护旧 HF revision |

Capture 与 Benchmark 在当前 QA500 **只包括** `core.l0` 和 `core.reconstruction_mse_token`。这两个是 per-token 后聚合，而非官方 SAEBench Core/官方 FVU/损失恢复率；严格使用 metric_id 区分。数学 scorer 采用可选 `math-verify==0.9.0`，缺失时不得称准确率已评估。Explain 的外部描述为 `source_annotations_only`，不宣称语义解释通过验证。原生 Circuits 的 transcoder feature ID 未必与主矩阵 SAE dictionary 共享坐标。

## 科学和工程硬性不变量

1. **输入真实**：每条题来自 pinned HF 源，question/context/choices 真实，参考答案只在本地 scorer 读取；model prompt 绝不包含 gold；不使用无来源随机合成问题。
2. **数据冻结**：seed、数据集配置和 split、抽样及 ID 哈希统一，跨模型不能换题；Hotpot/GPQA 等访问异常记录 blocked，不挑相似或更容易的题补上。
3. **模型真实**：`load_registered` 返回的模型、SAE、hook、layer、d_in、d_sae 必须实际匹配 registry。不得以只加载 HF weights/只运行 CLI dry-run 替代实际 forward。
4. **token 对齐**：输入由当前模型 tokenizer 编码；记录 prompt 和 token ID SHA；超过 2048 token 不静默裁剪，必须报错/部分状态。
5. **feature 真实**：激活特征在对应 token、对应 SAE 字典索引；非零激活、范围 [0,d_sae)、finite、排序，禁止猜 feature id 或跨 SAE 复用解释。
6. **干预真实**：`h_new=h+decode(z_edited)-decode(z)` 和真正 patch，维持 identity control；不改变当前存储的 SAE 模型或训练权重。
7. **解释来源**：Neuronpedia 需要 exact weight SHA256 + source 对照；缺失时保留 null。外部解释仅为文本来源，未通过独立 semantic/causal validation。
8. **恢复安全**：每个 run 单独 sample ID/配置/hash，禁止跨实验复用，追加成功行且拒绝重复或来源变化，失败写到独立 failure/receipt 文件，不假装 success。
9. **原生权限**：Circuits 需确实安装原生 tracer/兼容 transcoder；Train 用不包含 QA500 题目的独立训练集。两个子任务有额外授权和 native 结果核验。
10. **发布不可自动升级**：只有独立验收和人类 release authority 才可发布，保留历史 dataset 所有配置与 DOI/引用，不公开第三方题干及受限 GPQA 原文。

## 如何给本地 Agent 指派

先执行分支与环境预检。预检通过并核实模型/SAE/HF 访问权限后，才由操作者另行授权后续阶段。

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
bash scripts/experiments/qa500.sh full --allow-local-gpu
```

完整流程执行 source setup、预检、10 源冻结、20-site Pilot、全部 Core、全部 Effects、独立 Audit、HF staging；自动保留失败和 partial。若缺 GPU、HF 权限、源样本不足或模型不兼容，返回相应 blocked 收据而不是改协议。原生扩展是单独授权 `native --allow-native-compute`，HF publish 无任何一键隐含许可。

详尽命令、结果目录、恢复和 failure policy：`scripts/experiments/qa500/README.md`。

## 实验接收标准

- 软件：对确切提交运行 Python3.11/3.12 CI、Ruff、Harness 合同、数据适配/评分器负例测试。
- 数据：10/10 源有效、500 distinct real records each；明确访问/许可异常。
- GPU：20/20 模型层位点的真实 hook/SAE 与执行收据；真正失败如实保留。
- QA：50,000 baseline/model question 组合有对应模型 token/scorer 记录；数学 unsupported 分开。
- Core：100,000 SAE/site question 组合与 500 × 200 的唯一 ID，SHA/feature 数值校验。
- Intervene/Generate：4,000/1,000 小样本任务与 control/origin 完整。
- Explain/Circuits/Train：分别标记 source-only、native 原图和 native checkpoint，没有 source 时 blocked。
- HF：六个配置可 load_dataset，原 25 个配置和历史结果保持一致，清楚标注此为独立 cohort。

**目前软件分支的科学状态：CODE_PREPARED、CI_PENDING_OR_CHECK_SEPARATELY、GPU_UNRUN、HF_UNPUBLISHED。** 凡报告更高状态必须附实际 source commit、模型权重 identity 和独立审计。


## WP22 R2 架构迁移说明

已基于本轮审阅拆分 Dataset/Model/Task/Experiment/Runtime/Harness。项目结构权威为
[scripts/experiments/qa500/README.md](../../scripts/experiments/qa500/README.md)。

- 一级 `scripts/experiments/qa500.sh` 是唯一 Shell 入口。
- `conf/experiment/qa500/` 是科研配置所有者；`conf/experiment/qa500.yaml` 对接现有 Hydra experiment group。
- 十个源 Dataset 通过现有 SAEBase Dataset Registry 解析各自 `qa_adapter.py`。
- `models/qa_input.py` 负责输入，`tasks/capture` 与 `tasks/benchmark` 拥有通用算法。
- `scripts/experiments/qa500/` 只组合实例、执行矩阵、run-level summary 和研究发布准备。
- 所有 Task 实测写入现有 Runtime run leaf；跨 Run 冻结 cohort 与 Manifest 存在 `experiment/_studies/<protocol>/`。
- Hardware Profile 和隔离 Worker 是通用 Runtime；目前只允许重新初始化并用相同精度重试，其他 Offload 方法须经过真实 backend 验证。
- 当前版本 `assets.lock.json` 仍标注上游 SHA 未全部冻结，真实 GPU 和跨机器对比仍需独立验收，因此不能宣称达到零缓存跨机科学发布。

PR #111 的历史 QA500 平行模块已删除，禁止新的代码再引用 `saebase.datasets.qa_*`。


## WP22 R3 环境与结构整改（执行前强制门槛）

- 仓库根目录不得出现独立 `experiments/` 源码包；QA500 特定编排属于 `scripts/experiments/qa500/`，产品测试位于 `tests/workplan/wp22/`。本次将 14 个分散的研究编排模块按职责合并为 10 个，通用 Dataset/Model/Task/Runtime 保持在现有所有者中，不创建第二套 Registry。
- `qa-eval` 的 ANTLR extra 固定为 `math-verify[antlr4_9_3]==0.9.0`，与 Hydra Core 兼容。必须在相同 Python 环境实测 `pip check`、ANTLR 4.9.3、Math Verify 0.9.0 和真实评分负例。
- Agent 先确认当前仓库 `git remote -v` 和 `git config --get-all remote.origin.fetch`，若目标分支未在本地存在，使用 `git fetch origin refs/heads/fix/wp22-architecture-install-and-governance:refs/remotes/origin/fix/wp22-architecture-install-and-governance`，再显式切换。禁止修改现有 remote fetch 策略，禁止伪造成功。
- 依次检查项目 Python 版本、可用存储、包安装、导入/评分、冻结配置、CPU 负例、模型/SAE 实际下载权限与 GPU 资源。出现依赖解析失败时必须停止在 Preflight，返回直接冲突链和源码问题归因，而不是私自升级 Hydra、替换 scorer 或继续训练。
- `opensae_saebase.egg-info/` 为构建工具生成的辅助元数据，不是源码一级模块，也不代表版本发布。保持 Git 忽略并将临时安装/运行证据归入受控路径；如有已存在的用户数据不得自动删除。
- WP22 必须先通过独立安装与目录所有权检查，再由人工决定 Pilot 和正式大规模执行。HF Publish 必须另行授权。
