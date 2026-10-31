# WP21.2 本地 GPU Agent 后续工作单

**状态：待本地执行。** 本文件是 WP21.2 考试系统修复后的操作交接，不是另一套题目或评分标准。负责 Harness 的维护者已经在 PR #106 完成离线修复。下一步由有权重、GPU 与 round01 原始工作树访问权限的本地 Examiner 执行。

- Harness 修复分支：harness/wp21-2-hardening-20261030，PR #106
- 冻结试卷提交：8a78be1209bb30c652d500f934c226a5d54684c9
- 冻结 SAEBase 产品基线：41a54cbfdee5cce24d3d8f7517aadef0c752c5ac
- round01 历史归档：PR #105，17 PASS、R08 ENV_BLOCKED、R17 FAIL、R20 INCOMPLETE
- 基本执行手册：harness/tests/wp21/WP21_2_HARDENING.md
- 冻结题卡：harness/tests/wp21/cases/R08.md、R17.md、R20.md

## 一、直接交给本地 Agent 的任务

作为 **WP21.2 Local Examiner**，从上述 Harness 分支检出可信工具，独立完成本工作单的 F00–F05。你负责发现问题、真实执行、保留证据和清晰归因；不负责修改冻结题目、Contract、Harness、评分器、历史 review_final、Atlas 或 main。

先在已有 round01 目录中定位真实候选工作树、外置 control 和独立 reviewer 目录，再执行资源预检。所有原始目录保持只读；任何需要产生新 checkpoint、运行结果、审查输出或诊断数据的命令，写到新的 round02 或 followup 专用目录。保留所有失败的原始命令、退出码、日志和不满足容差的数值。禁止 toy 或随机权重冒充正式实验，禁止把新版本修复结果倒填到 round01 的历史成绩。

**交付目标**：一份逐项结果表、环境和资产 identity、R08 新独立验收、R17 训练可复现性原因、R20 审计复核、Benchmark 问题证据、Reviewer 溯源档案的完整性索引。没有本地原始证据时明确记录 MISSING_LOCAL_EVIDENCE，不得编造结论。

## 二、必须先找到的本地资源

以下是路径变量的含义，**不预设任何路径已经存在**。Examiner 先查找、核对绝对路径，随后仅在自己的 shell 环境赋值。

| 变量 | 实际资源 | 缺失时怎么办 |
| --- | --- | --- |
| GEMMA_SNAPSHOT | 已缓存的 google/gemma-2-2b 冻结预训练 snapshot | 记录 ENV_BLOCKED |
| SAE_SNAPSHOT | Gemma Scope layer25、blocks.25.hook_resid_post 对应的只读 SAE snapshot | 相关推理题记录 ENV_BLOCKED |
| PYTHIA_SNAPSHOT | EleutherAI/pythia-14m，revision d31f659f077f9ea500c4d7faa999893d7f28cbdf | 环境准备者在开考前获取并锁定；R08 不得在线下载 |
| ROUND01_ROOT | round01 调度目录，含 candidates/Rxx、control/Rxx 和 dispatch.json | 不能原地复核，标 MISSING_LOCAL_EVIDENCE |
| ROUND01_REVIEWS | round01 的 Reviewer 目录，含 Rxx/review.json 和原始 records | 不能验证历史独立脚本，标 MISSING_LOCAL_EVIDENCE |
| FOLLOWUP_ROOT | 全新的工作与诊断输出目录，必须在 candidate 与可信源码之外 | 先新建 |
| R17 weights / plans | 两个 seed=7 的真实 native checkpoint、plan.json，另有 seed=11 的 checkpoint、训练日志、数据身份 | 不能做训练复现归因，保留原 FAIL |

历史汇总里记录的环境是 Python 3.12.14、torch 2.11.0+cu128、RTX 5070 12GB，Gemma revision c5ebcd40d208330abc697524c919956e692655cf，SAE revision fd571b47c1c64851e9b1989792367b9babb4af63。这些仅是 **round01 报告声称的值**；必须通过本地实际路径、文件 SHA256、锁和运行探测重新核对，不能从名称推断身份。

## 三、F00 资源与可信工具门禁

从 WP21.2 的可信 checkout（不是 candidate）执行：

    python3.12 harness/tests/wp21/tools/manage.py check-spec
    python3.12 -m pytest -q harness/github_work_test/test_wp21_local_suite.py harness/github_work_test/test_wp21_exam_ready.py harness/github_work_test/test_wp21_hardening.py

    python3.12 harness/tests/wp21/tools/preflight.py --all \
      --model-snapshot "$GEMMA_SNAPSHOT" \
      --sae-snapshot "$SAE_SNAPSHOT" \
      --pythia-snapshot "$PYTHIA_SNAPSHOT" \
      --output "$FOLLOWUP_ROOT/preflight_all.json"

预检有 ENV_BLOCKED 时工具允许非零退出码；保存 JSON 和实际 stderr，再处理缺失。它只检查资源存在性，不代表模型已经真正前向执行。

R08 的资源由可信环境准备者**在考试进程之外**使用官方 Hugging Face Hub 以固定 SHA 获取。需要提供 revision、实际 checkpoint/config/tokenizer 字节哈希与下载来源；考生和独立 reviewer 必须离线。R08 不要求 SAE，Pythia 资源使用独立资产锁：

    python3.12 harness/tests/wp21/tools/preflight.py \
      --model-snapshot "$PYTHIA_SNAPSHOT" \
      --freeze-assets "$FOLLOWUP_ROOT/r08_assets.lock.json"

    python3.12 harness/tests/wp21/tools/probe.py --case R08 \
      --model-snapshot "$PYTHIA_SNAPSHOT" \
      --assets-lock "$FOLLOWUP_ROOT/r08_assets.lock.json" \
      --device cpu --output "$FOLLOWUP_ROOT/r08_probe.json"

只有 probe 报告 REAL_MODEL_PROBE_OK_NOT_EXAM_PASS、model_executed=true，且实际 architecture=GPTNeoX、6 层、宽 128，并通过权重来源核对，才进入 F01。probe 自身 **永远不是 R08 PASS**。Gemma/SAE 的资产锁和实际加载探测仍按 EXAM_READY.md 执行，必须对应本地实际缓存而非随意指定目录。

## 四、F01 R08 新 pretrained Model Provider 重考

- 环境准备完成后，用可信 manage.py 为 R08 创建 **新 run_id 和新目录**。如果调度者有真实用户授权，记录其来源；没有则只准备，不伪造 assignment。
- 只给考生 R08 题卡、prompts/R08.md、允许路径、只读 Pythia snapshot 和独立工作树。考生实现 GPTNeoX hook 适配、显式 external provider 注册、真实前向 capture 与 identity patch。
- 使用独立 reviewer 再跑冻结的 **5 个正例、4 个负例**，比对真实张量、旧 HF backend 兼容以及未知 hook/重复 provider 的拒绝行为。
- 收集 source diff、资源锁、输入、全部命令日志、实际 tensor 数值和 Reviewer asserts。不能把预检成功视作题目成功。

**交付**：新的 R08 review_final、对应 Reviewer 原始命令和 assertions、模型身份与 sha 清单、完整交付文件索引。历史 R08 ENV_BLOCKED 原封不动保留。资源仍然缺失则交实际 preflight blocker，并停止该题。

## 五、F02 R17 同 seed SAE 训练诊断和新复核

此题目前有两个独立缺口：同 seed 的 W_dec 差异（历史 max_abs 约 0.083），以及 native_training_logs/ 被旧审计器误判。应分别诊断。

1. 首先用 WP21.2 的 audit_evidence.py **只读复核** round01 R17 的原候选目录及外置 control。输出新的 followup 审计 JSON；不得修改原 candidate evidence 或旧审查结论。如果历史训练日志目录存在且有真实可哈希文件，记录审计修复前后的差异。
2. 收集 seed=7、7、11 三次训练原始日志、完整 plan、训练语料 sample/group IDs、原生配置、checkpoint、CUDA/torch/SAELens 版本、真实训练 token 数。原始数据缺失要显式登记。
3. 对两个真正的 seed=7 权重运行当前训练差异诊断工具。下面的命令以实际文件路径替换，输入应为原始真实输出：

       python3.12 harness/tests/wp21/tools/training_repro.py \
         --weights-a /actual/seed7_a/sae_weights.safetensors \
         --weights-b /actual/seed7_b/sae_weights.safetensors \
         --plan-a /actual/seed7_a/plan.json \
         --plan-b /actual/seed7_b/plan.json \
         --rtol 0.0001 --atol 0.00001 \
         --output "$FOLLOWUP_ROOT/r17_comparison.json"

4. 排查 seed 是否进入 Plan、native runner 和 checkpoint 身份；对照 CPU/CUDA RNG、数据加载及顺序、torch deterministic 设置、CuBLAS workspace、cuDNN/TF32、dtype 和 TopK kernel。报告 **第一个可定位的差异来源**，区分 baseline 产品实现、候选改动和运行环境。
5. 如需证明修复有效，另建新 attempt，执行 seed 7/7/11 三次真实训练，每次 1024 token、总量不超过 3072，随后真实 heldout encode/decode 和冻结正反例重放。新的结果独立评分，旧的 57 分及 FAIL 不得修改。

**交付**：R17 原始资产索引、两次权重对照统计（包括 failing_elements、max_abs、dtype、checkpoint hash）、seed/随机性和数据顺序溯源、单独列出的目录审计结果，以及新的 attempt / 环境阻塞结论。保持 rtol=1e-4、atol=1e-5。

## 六、F03 R20 目录审计复核

- 找到原 candidate 的 benchmark_output/ 真实目录、原始 metrics 和目录内可哈希文件；使用旧 evidence/manifest 对照，不得只看目录名。
- 用新的 audit_evidence.py audit 对原工作树执行只读复核，输出到 FOLLOWUP_ROOT；只允许新的诊断 JSON，不重写原历史报告。
- 原 round01 的 5/5 正例和 4/4 负例被报告为独立复跑完成。必须找到对应的 Reviewer scripts、assertions、命令原文、实际产物和 upstream hashes 才能充分核查；找不到则保留 **历史报告成立、原始证据缺失** 两种不同事实。
- 若记录足够且确需重新出具通过结论，必须建立新的 Reviewer attempt，重新完成所有冻结检查和独立 finalize；**不能修改原 round01 INCOMPLETE 的 review_final**。

**交付**：原审计失败原因、新审计状态、目录文件清单及 sha、是否具备独立复跑证据、必要时的新 R20 审查结果。仅凭新 audit 不得宣布自动 PASS。

## 七、F04 Benchmark 真实设备与 padding 诊断

检查生产代码 saebase/benchmarks/runner.py、saebase/interventions/engine.py 和 tasks/benchmark/backend/runner.py 的真实调用链：

1. 同一真实模型/SAE 对 **至少两条不同 token 长度** 的输入，记录 tokenizer 产生的 token IDs、pad_token_id、padding side、BatchEngine 实际接收的参数、真实模型输出与异常。
2. 在相同真实 checkpoint 上，记录配置期望设备、load_registered 实际设备、模型参数所在 device、capture/intervention 张量 device、CUDA 可用情况；不能因为配置写 cuda 就宣称 GPU 运行。
3. 先复现，再归因。如果确定产品问题，单独提交**产品级修复候选**与回归用例，不准通过改 Harness 评分条件使历史结果通过。请报告精确模块、函数、输入、观察差异及最小修复建议，不要未核验就修改冻结 SAEBase 基线。

**交付**：两条不同长度输入的真实对照日志、padding/device 传播证据、最小复现命令、修复是否必要及影响范围。涉及产品 PR 须与 Harness PR 分离。

## 八、F05 Reviewer 原始证据溯源

依次检查 round01 的 R01–R20 Reviewer 工作树。每题如有原 review.json、review_final.json、独立脚本、assertions、stdout/stderr，使用 WP21.2 工具建立**仅本地保存**的归档并立即 verify：

    python3.12 harness/tests/wp21/tools/archive_review.py create \
      --review /actual/round01_reviews/R01 \
      --output "$FOLLOWUP_ROOT/reviewer_archives/R01_review.zip"
    python3.12 harness/tests/wp21/tools/archive_review.py verify \
      --archive "$FOLLOWUP_ROOT/reviewer_archives/R01_review.zip"

对每题分别执行，不要用一份脚本或汇总 JSON 冒充 20 份独立执行记录。记录 archive SHA256、Reviewer 脚本数量、检查覆盖数及缺失项。原始日志可能暴露私有路径或敏感环境，**默认不把原始 archive、checkpoint、token、绝对路径、实验大张量推送到 GitHub**。任何发布前必须人工检查与必要脱敏。

17 个历史 PASS 的再次审计优先核查脚本确实位于考生工作树之外、是否调用真实生产接口、有无独立实际计算和负例退出码。发现缺口记为新诊断，不倒填历史状态。

## 九、结果格式与停机条件

在 FOLLOWUP_ROOT 内写独立的 followup_report.md 与 followup_manifest.json，至少包含如下字段：

- tool_branch、tool_commit、exam_commit、product_base_commit、run_id、实际执行日期和环境版本；
- F00–F05 每项状态：DONE、BLOCKED、MISSING_LOCAL_EVIDENCE、FAILED，实际命令和退出码、精确 blocker；
- R08/R17/R20 的旧状态、新 attempt 状态（如有）、逐项正反例计数、真实 asset/source SHA 和路径的**脱敏相对引用**；
- R17 容差与权重差异、R20 审计目录前后对比、Benchmark 实际 device/padding 对照；
- 各 Reviewer provenance archive 的本地 SHA256、存在/缺失数量、独立测试脚本的可核查来源；
- 按 CONTRACT_GAP、SELECTION_GAP、IMPLEMENTATION_ERROR、ENFORCEMENT_GAP、EVALUATOR_SCOPE_GAP、ENV_BLOCKED 记多原因并标责任方；
- 哪些问题仅需 Harness 工具更新、哪些是产品实现缺陷、哪些需要更多实物证据，列出可复现的最小反例。

不得删除或覆盖原 round01。若任一资产来源无法核验，拒绝将其用作真实模型 PASS；若缺少原始工作树，报告明确缺口并保留历史归档。发布到 GitHub 的仅是经审核、脱敏的报告与必要代码补丁；原始权重和未经审核的日志保存在可信本地。

## 十、完成标准

- F00 所有 20 题资源要求被核对，并明确 blocked 项目；
- R08 有新的真实独立评审，或明确的权重缺失与其他 blocker；
- R17 同时有训练数值差异诊断与目录审计的分离结论，不改旧容差；
- R20 原目录证据得到新审计结论，明确是否足以重新独立验收；
- Benchmark padding/device 有生产调用链的真实结果，而非仅查配置；
- Reviewer R01–R20 的档案逐题可验证或逐题明确缺失；
- 结果可追溯，原始 round01 记录不变，所有新增产物仅落在新目录。

这份交接单只规定下一轮 Examiner 的操作和证据边界，**不新增或修改 WP21.1 的考试正反例**。如需修改 Harness 规则与评分器，由 Harness 维护者根据实际发现另开版本处理。