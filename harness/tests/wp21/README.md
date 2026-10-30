# WP21 真实 SAEBase 综合测试

**2026-10-30 WP21.1 更新：考试执行工具链与独立 Review 流程请先读 [Exam Ready 手册](EXAM_READY.md)。** 冻结20道题不变，离线工具就绪并不表示20个真实模型考试已经完成。

本轮为 20 个独立 agent 准备 20 道真实仓库任务。基线为 `41a54cbfdee5cce24d3d8f7517aadef0c752c5ac`。试卷、工具与本地规范已提供；逐题真实 checkpoint 实验的初始状态为 `NOT_RUN`。

从 [审计](AUDIT.md)、[通用执行协议](../../contracts/REAL_WORKFLOW_PROTOCOL.md) 和自己的一张 [题目卡](cases/) 开始。完整机器规格在 [cases.json](cases.json)，Work Plan 在 [21_real_saebase_workflow_exams.md](../../github_work_plan/21_real_saebase_workflow_exams.md)。考生按题目限定上下文读取源码。

## 1. 本轮与旧考试的区别

每题都要使用真正的 SAEBase 生产模块。Task 题使用实际 Registry、Runner、配置和任务实现；模型扩展题使用公开 Extension API；文件管理题操作真实 SAEBase 产生的实验目录。`solve(payload)`、配置打印、toy model、写入一个声称 PASS 的 JSON 都不能满足正式 checkpoint 验收。

正式七类任务通过 `python -m saebase.runtime.train` 进入。默认 `task.execute=false` 只完成配置阶段。真实运行显式传 `task.execute=true`。原有 `saebase` CLI 是兼容入口，其支持范围和参数不能直接替代本轮入口。

R08 使用真正新增的 Pythia14m model-only 扩展；R09 只删除试题自带的临时模型 provider。二者可使用 model-only 证据，`sae_identity=null`。其他需要 SAE 的题目记录实际原有或新训练 SAE 身份。新增实验组件不自动纳入正式 Atlas，也不扩大公开七类 Task 的产品承诺。

## 2. 交付目录

| 内容 | 位置 |
| --- | --- |
| 不可变试卷 | `harness/tests/wp21/cases.json`、`cases/Rxx.md` |
| 共用输入和删除题起点 | `harness/tests/wp21/fixtures/` |
| 产品代码 | 每题 `allowed_paths` 指定的 `datasets/`、`models/`、`tasks/`、`conf/`、`saebase/` 文件 |
| 考生运行脚本与回归 | `tests/wp21/Rxx/` |
| 考生交付与命令日志 | `harness/tests/wp21/runs/<run_id>/Rxx/` |
| SAEBase 真实产物 | 当前工作树的 `experiment/<dataset>/<model>/<task.path>/<sub_name>_<seed>/` |
| 独立验收 | 工作树外的 `control/Rxx/` 与 reviewer 自己的工作目录 |

已有 Runtime 的单数 `experiment/` 保持原义。准备工具预建它；题目工作树拥有该输出根。所有来源和产物都使用当前 case 的工作树，避免不同 agent 共用可写代码或叶子目录。

## 3. 调度员的准备流程

以下命令由可信调度员从未被考生修改的本包源码执行。它们只做本地操作。

```bash
python harness/tests/wp21/tools/manage.py check-spec
python harness/tests/wp21/tools/manage.py prepare --destination /absolute/wp21_run --run-id round01
```

默认生成 20 份独立源码副本和逐题固定初始化材料，状态为 `prepared_not_assigned`，不运行实验。单题使用 `--case R02`。目标目录必须尚未存在，且位于本包源码之外；所有删除题的已有功能由该工具自动安装。

用户已经明确把题目分派给 agent 后，调度员可在**生成新工作树时**附带 `--assignment-ref`，写入该实际指令的来源或原文摘要。例子中的说明必须换成真实已有的分派依据。

```bash
python harness/tests/wp21/tools/manage.py prepare --destination /absolute/wp21_assigned --run-id round01 --case R02 --assignment-ref '用户已明确分派 WP21 R02，仅在本地执行'
```

这个记录承接已有的用户指派，作用于冻结的 case 范围；它不伪造 Atlas、人类 PR review 或正式发布授权。考生不得创建、修改 `.cursor/wp21_assignment.json`，也不得自行运行 dispatcher 的准备或授权参数。仅要求“准备题目”时保持未分派状态。

准备工具生成 `candidates/Rxx/` 和外置 `control/Rxx/`。用户可直接把 [总调度提示](prompts/00_dispatch_prompt.md) 交给负责安排工作的 agent；调度员根据实际分派做这些机械步骤。无需二十次填写人工逐文件表格。

## 4. 环境与权重

Python 使用项目要求的 3.11 或 3.12。推理依赖依据 `pyproject.toml` 的 analysis/tl4 extras，训练题使用已含项目所调用 API 的 SAELens v6/TransformerLens v4 环境。依赖环境在源码工作树外由调度员提前准备，安装后记录**实际**版本；本包的 CPU 准备检查只证明随包记录的环境，并未验证任意依赖组合。候选工作树使用源码导入，不在freeze之后执行editable install或构建wheel；这样安装产生的egg-info/build文件不会混入产品diff。录制器设置PYTHONPATH为当前候选源码根，种子脚本也显式加入其ROOT。

在各工作树根运行环境检查，例如

```bash
python harness/tests/wp21/tools/preflight.py --case R02 --model-snapshot /absolute/gemma_snapshot --sae-snapshot /absolute/gemma_sae_snapshot
```

预检不下载权重，不把“目录存在”当作模型成功加载。先由调度员准备一次共享只读缓存，冻结实际 snapshot revision、配置、tokenizer 与权重 SHA，再分派计算任务。Gemma checkpoint 使用已经取得访问资格的来源。模型缺失、依赖缺失或设备不可用时记 `ENV_BLOCKED`，不把 toy 替代品写成真实实验。

真实 Gemma 与 SAE 身份分别是 `google/gemma-2-2b`、`gemma2_2b_pt_l25_canonical`，默认 layer 25、`blocks.25.hook_resid_post`。具体不可变上游 revision 和实际文件哈希由现场缓存冻结，不以 `main`/`latest` 冒充。`preflight --model-snapshot/--sae-snapshot` 只检查或哈希目录，不会把任意路径接入 `load_registered`。调度员须为本轮选择独立、只读的 Hugging Face 缓存，并在进程环境固定 `HF_HOME`/`HF_HUB_CACHE`；其中对应模型 ID 的 refs、snapshots 和 blobs 必须解析到已经冻结的 revision。保留原共享缓存状态，实际运行再核对 loader 身份及加载后权重，不凭预检路径推断采用了该 checkpoint。R08 的 Pythia14m revision 已在题卡固定。

调度员可一次计算共享资源锁。该文件包含私有本地路径，保存在外置控制目录，不提交进源码。

```bash
python harness/tests/wp21/tools/preflight.py --model-snapshot /absolute/gemma_snapshot --sae-snapshot /absolute/gemma_sae_snapshot --freeze-assets /absolute/wp21_run/control/assets.lock.json
```

20 个 agent 可独立改代码。默认同一 GPU 同时只安排 1 个模型作业，CPU 作业最多 4 个，每个作业遵守题卡预算。调度是简单排队，不需要另建调度系统。权重不复制进 20 份工作树，也不放进交付 ZIP。

## 5. 每个考生怎样工作

先读自己的 case、合同选择及少量指定上下文，写下预期代码变化与相关负例。基线缺陷可以是题目目标，也可能是新发现；按事实记录归因。删除题先运行 `before.py --check-only`，再按题卡运行真实 before，保存不可变证据。`--check-only` 不计正式运行通过。

实现写回实际产品模块，回归脚本放本题 `tests/wp21/Rxx/`。可读的原题与评分要求保持冻结。每次运行用新叶子；`+experiment.overwrite=false` 可显式增加原配置未声明的保护键，`experiment.overwrite=true` 不用于本轮。不要把清空原目录后重跑称为恢复。

命令录制器实际启动子进程，并保存 argv、退出码、超时及日志哈希。下面的 R02 运行脚本由考生按题实现，示例不表示该脚本已存在。

```bash
python harness/tests/wp21/tools/record.py --cwd /absolute/wp21_run/candidates/R02 --control /absolute/wp21_run/control/R02 --output /absolute/wp21_run/candidates/R02/harness/tests/wp21/runs/round01/R02/commands/0001.json --timeout 600 -- python tests/wp21/R02/run_case.py
```

录制器设置 Hugging Face 离线环境变量，以使用已准备缓存。这些变量不构成 OS 网络隔离。独立验收需要的只读挂载、低权限运行和网络控制由执行环境负责。

## 6. 证据文件

调度员模板均为 `NOT_RUN`，不会通过验收。考生填写 `evidence.json` 和 `contract_receipt.json`，保留实际命令记录、自测、源码预期、候选 Context 与总结。结构详见 [证据字段](EVIDENCE.md)。每张卡另列本题原始数组、对照或 ledger。

`contract_receipt.json` 使用 **`saebase.wp21.contract_receipt.v1`**，合同目录版本继续是 `1.1.0`。旧 `harness.contracts.trace` 和 `acceptance` 针对仅允许 Harness 沙箱编辑的历史考试；WP21 使用自己的范围与收据验证，历史校验保持原样。

调度员或独立 reviewer 从可信工具副本收集实际 diff 和产物指纹

```bash
python harness/tests/wp21/tools/audit_evidence.py collect --workspace /absolute/wp21_run/candidates/R02 --control /absolute/wp21_run/control/R02
python harness/tests/wp21/tools/audit_evidence.py audit --workspace /absolute/wp21_run/candidates/R02 --control /absolute/wp21_run/control/R02 --output /absolute/wp21_run/control/R02/evidence_audit.json
```

`collect` 填充 `source_manifest.json`、`diff.patch`、`artifacts_manifest.json`，返回实际 source hash，供考生填到 `evidence.json`。`audit` 只可能确认 `EVIDENCE_READY_FOR_REPLAY`，不输出行为 PASS 或分数。其职责是识别状态、身份、范围、缺文件、损坏或不一致；独立复跑仍按 [验收规程](REVIEW.md) 完成。

## 7. 失败和交付

考生报告 `SELF_TESTED`、`BLOCKED` 或未完成项，独立 reviewer 写最终逐项判定。沿用已有六类失败归因，另标来源是 baseline、candidate、环境、治理或评分器。失败 trial、错误原文与旧产物保留。缺少设备只报告阻塞，不自动判 agent 算法实现错误，也不记真实实验通过。

本轮只交本地文件。远程仓库、PR、release、网站与 Atlas 权威状态各自保留原有授权边界。
