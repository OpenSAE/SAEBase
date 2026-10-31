# WP21.1 真实综合考试执行手册

> WP21.2 的独立修复与后续本地 GPU 验证说明见 [WP21_2_HARDENING.md](WP21_2_HARDENING.md)。该版本不修改冻结题目或已完成考试的历史判定。

**Exam-system ready** 指 20 题冻结材料、隔离工作树生成器、环境阻塞检查、命令记录、独立 Reviewer 复跑入口和汇总程序完成并通过离线回归测试。**Real-model execution ready** 另需实际的 Python 3.11/3.12、完整依赖、模型及 SAE checkpoint、必要 GPU 与网络隔离。**Case PASS** 只能在对应 Agent 完成工作、独立 Reviewer 复算所有正反例后产生。首次交付本包时 20 题均为 NOT_RUN；如今已有 round01 与 round02 的历史结果，具体见 [阶段结项](../../github_work_plan/21_harness_phase_closeout_20261030.md)。本手册中的示例仍为新考试的准备操作，不代表自动通过。

## 一、需要遵守的身份边界

- 教师工具与控制目录由可信调度员保管。每位 Agent 只拿到一份 `candidates/Rxx/` 工作树、自己的题卡和提示、资源路径及只读控制记录。Agent 无权修改 `harness/`、`.cursor/`、冻结 case 或外部 control。
- `suite_lock.json` 锁定 68 个题目/Contract/提示/Fixture 文件的原始字节。`manage.py check-spec` 同时核对 20 题合同、范围、种子和该锁。锁本身应使用交付 ZIP 的独立 SHA256 确认，不能从不可信的候选目录重新生成。
- 现有 `catalog.json` 22 个 Contract、SAEBase 产品实现、原 Runtime 和 20 道题的目标不变。此次 WP21.1 只增加 Harness 考试与审查能力。
- `record.py`/权限 Hook 是逻辑安全措施，不是 Linux 容器隔离。真实调度仍需只读 checkpoint/缓存、无网络执行、受限写权限、GPU/CPU 并发限制。GPU 默认每卡一个模型进程，CPU 同时最多四个工作进程。

## 二、教师准备与检查

从 ZIP 里的**可信源码副本**调用工具，绝不能运行 Agent 改写的裁判工具。

```bash
cd SAEBase_WP21_Local/SAEBase
python3.12 harness/tests/wp21/tools/manage.py check-spec

# 根据每题配置检查本地环境；不自动下载权重
python3.12 harness/tests/wp21/tools/preflight.py --case R01 \
  --model-snapshot /readonly/hf/gemma_snapshot --sae-snapshot /readonly/hf/sae_snapshot

# 模型/SAE资源指纹存到候选工作树外
python3.12 harness/tests/wp21/tools/preflight.py \
  --model-snapshot /readonly/hf/gemma_snapshot \
  --sae-snapshot /readonly/hf/sae_snapshot \
  --freeze-assets /absolute/wp21/control/assets.lock.json
python3.12 harness/tests/wp21/tools/preflight.py \
  --verify-assets /absolute/wp21/control/assets.lock.json

# 在具备依赖和预先缓存真实权重的隔离执行环境中，真正加载并前向
python3.12 harness/tests/wp21/tools/probe.py --case R01 \
  --model-snapshot /readonly/hf/gemma_snapshot \
  --sae-snapshot /readonly/hf/sae_snapshot \
  --assets-lock /absolute/wp21/control/assets.lock.json \
  --device cuda --output /absolute/wp21/control/model_probe.json
```

Gemma+SAE 的实际 `load_registered` 缓存解析仍须核验到已冻结的 blobs/revision；探测脚本不会把传入路径自动绑定给该 loader。R08 走自己的 Pythia14m model-only profile；R09 是 Gemma model-only；R17 在真实训练前仅需 Gemma model-only 预检，生成的 SAE 身份由独立训练验收获得。训练题要求单卡 CUDA，其他题按真实设备能力执行。

未满足依赖、路径、权限、缓存或设备条件时记录 `ENV_BLOCKED`，不要以 toy/随机权重替代真实试卷。

## 三、分派独立考生

用户已明确授权执行对应题目后，调度员生成**新的**目录。这里的 `--assignment-ref` 必须引用真实已有的用户授权，而不是示例文字。

```bash
python3.12 harness/tests/wp21/tools/manage.py prepare \
  --destination /absolute/wp21_round01 --run-id round01 \
  --assignment-ref '来自实际用户指令的授权来源'
```

得到 `candidates/R01..R20/`、`control/R01..R20/`；未显式授权时省略 `--assignment-ref`，状态为 `prepared_not_assigned`，并且录制器会拒绝执行考试。每个考生只阅读自己的 `prompts/Rxx.md` 和卡片，不得编辑教师控制文件。五道删除题的初始化种子自动安装，每题独立。

候选命令必须带外置控制目录，且 timeout 不能超过该题剩余累计时间预算。

```bash
python3.12 harness/tests/wp21/tools/record.py \
  --cwd /absolute/wp21_round01/candidates/R02 \
  --control /absolute/wp21_round01/control/R02 \
  --output /absolute/wp21_round01/candidates/R02/harness/tests/wp21/runs/round01/R02/commands/0001.json \
  --timeout 600 -- python3.12 tests/wp21/R02/run_case.py
```

录制器保存真实退出码、超时、日志 hash、assignment 和累计录制耗时。每次用新的四位递增命令编号，保留失败调用；不能由 Agent 删除日志重新清零。`--control` 目录只能由可信调度员提供并保持只读。prompt/trial/token 数预算仍需要 Reviewer 依据真实执行数组、trial ledger 和训练日志复算；工具层无法凭命令行精确推断它们。

在 Agent 实际填写 `evidence.json`/`contract_receipt.json` 后，教师调用：

```bash
python3.12 harness/tests/wp21/tools/audit_evidence.py collect \
  --workspace /absolute/wp21_round01/candidates/R02 \
  --control /absolute/wp21_round01/control/R02
python3.12 harness/tests/wp21/tools/audit_evidence.py audit \
  --workspace /absolute/wp21_round01/candidates/R02 \
  --control /absolute/wp21_round01/control/R02 \
  --output /absolute/wp21_round01/control/R02/evidence_audit.json
```

`EVIDENCE_READY_FOR_REPLAY` 仅表示材料完整，不等于行为通过。`required_artifacts`、全量命令记录、可信 assignment、累计时间、产物和来源 hash 均会核验。

## 四、独立裁判

```bash
python3.12 harness/tests/wp21/tools/review.py init \
  --candidate /absolute/wp21_round01/candidates/R02 \
  --control /absolute/wp21_round01/control/R02 \
  --destination /absolute/wp21_reviews/R02
```

它从干净基线重建 reviewer 工作树，按冻结 seed 初始化，仅移植候选**生产代码**改动，排除考生 `tests/wp21/R02/**` 的测试脚本。Reviewer 在自己控制的目录下建立独立测试脚本，使用生产入口实际加载 checkpoint 并复算 raw tensors、registry 变化、数值、trial ledger、产物路径/哈希或阶段依赖。可从 `behavior.py` 复用五类纯验证函数。

该题冻结正例依序编号 `P01..`，负例编号 `N01..`。独立脚本在进程参数中接收 `--wp21-assertions-output` 与 `--wp21-check-id`，在 reviewer 记录目录生成 JSON，例如：

```json
{
  "schema": "saebase.wp21.reviewer_assertions.v1",
  "check_id": "P01",
  "assertions": [
    {"name": "独立实际 capture 行数与 sample ID 检查", "passed": true,
     "observed": {"count": 8, "source": "independent raw replay"}}
  ]
}
```

这个示例只是**格式**，绝不可复制作为实际 PASS 证据。`passed` 必须由 reviewer 独立计算产生，并保留实际命令、模型导入路径、原始数值、异常日志和 checkpoint identity。不同题目可用不同测试脚本，但检查标准必须来自已冻结的题卡。

```bash
python3.12 harness/tests/wp21/tools/review.py run \
  --review /absolute/wp21_reviews/R02 --check P01 \
  --timeout 600 --assertions yes -- \
  python3.12 /absolute/wp21_reviews/reviewer_scripts/R02_P01.py

# 每项P/N独立验证后，汇总证据/资源/全部硬门槛
python3.12 harness/tests/wp21/tools/review.py finalize \
  --review /absolute/wp21_reviews/R02 \
  --assets-lock /absolute/wp21_round01/control/assets.lock.json \
  --probe-report /absolute/wp21_round01/control/model_probe.json
```

负例若预期触发运行错误，Reviewer 的独立脚本应确认该目标子过程确实失败、未写 completed，并以 reviewer 检查程序自身退出码 `0` 代表该条负例验证通过。`review.py run` 不会仅凭非零退出码自动判断负例成功。每个正反例均需独立 `PASS_VERIFIED`；独立模型探测、真实候选证据、范围和所有检查全部满足才能给最终 `PASS`。只做一两条检查的合成演示只能得到 `INCOMPLETE`。

Reviewer 如遇失败，在 `finalize` 可选 `--attribution` 六类与 `--owner` 记录归因。对失败、环境阻塞及未完成都保留原始记录，不伪造人类审批。

## 五、全轮成绩表

```bash
python3.12 harness/tests/wp21/tools/summary.py \
  --dispatch /absolute/wp21_round01/dispatch.json \
  --reviews-dir /absolute/wp21_reviews \
  --output-dir /absolute/wp21_summary_01
```

得到 `summary.json` 和 `summary.md`。无最终 reviewer 文件的题目仍为 `NOT_RUN` 或 `ASSIGNED_NOT_REVIEWED`，分数为空。只有独立最终判定才能显示 `PASS`。

## 六、各验证器的边界

- `behavior.py` 的 numerics/intervention 仅在获得**实际** activation/codes/logits 和重算输出后可证明对应数学不变量。
- 组件生命周期要求实际 before/after 不同进程的 Registry/Runner 查询；传入手写的 ID 列表不会证明真实组件退役。
- grid/search/resume 校验试验身份、组合、均值、历史 hash；实际 trial 对生产 API 的调用仍需命令/原始产物证明。
- file/pipeline 验证器核对真实字节、路径约束和阶段 hash；不替代真实模型执行。
- `probe.py` 只是前置模型载入能力探测，不能代替每题正负检查，也不能从缓存目录名称直接推断 loader 实际使用的 revision。

本升级包按工具链就绪验收，环境和真实考生工作仍需后续在具备相关权重/设备的执行环境完成。正式 GitHub、Atlas 审批和发布边界不变。
