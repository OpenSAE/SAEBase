# SAEBase Harness · WP21 阶段结项（2026-10-30）

## 结论与统一入口

**WP21.1 考试系统和 WP21.2 Harness 修复阶段结项。** 从此以本仓库 \`main\` 上的 Harness 为后续工作起点；WP18、WP19、WP20 保留原有题目、验证器、结果与权限边界。WP21 的冻结试卷、68 项文件锁和 22 个通用 Contract 继续有效。只有新考试轮次可以生成新成绩；历史原始记录保持不可变。

- [Harness 总入口](../README.md)
- [WP21 题目与工具](../../tests/wp21/README.md)
- [WP21.1 Exam Ready 手册](../../tests/wp21/EXAM_READY.md)
- [WP21.2 修复记录](../../tests/wp21/WP21_2_HARDENING.md)
- [WP21.1 round01 报告](21_exam_round01_report_20261030.md)
- [WP21.2 round02 后续复核](21_wp21_2_followup_report_20261030.md)
- [后续 GPU 工作单](21_wp21_2_local_gpu_followup.md)

## 阶段范围

| 版本 | 已结项内容 | 不应混同的结论 |
| --- | --- | --- |
| WP18 | Agent Flow、权限、任务合同、独立审核与沙箱测验 | 与真实科研模型完整能力不同 |
| WP19 | 200 道微型任务、Contract 绑定和 Reference/Verifier 校准 | 合成任务通过不等于真实生产运行 |
| WP20 | 20 道集成题的独立 Reference 与正反例；历史结项保留 | 旧实验验收级别不提升 |
| WP21.1 | 20 个真实 SAEBase 工作流独立考生、冻结题目、隔离调度、Reviewer 和历史成绩 | 只说明对应冻结硬门槛 |
| WP21.2 | 目录交付审计、资源预检、模型探测、Recorder 负例记录、Reviewer 溯源归档、复现诊断与多原因说明 | 未修改原评分器、题目容差和历史成绩 |

### 真实考试记录

- round01（冻结考试提交 \`8a78be1209bb30c652d500f934c226a5d54684c9\`）：**17 PASS、1 ENV_BLOCKED（R08）、1 FAIL（R17）、1 INCOMPLETE（R20）**，逐题原始最终判定在 \`harness/tests/wp21/runs/round01/reviews/\`。
- round02（独立后续验证，仅重新考试 R08）：**R08 PASS 100，5/5 正例、4/4 负例**。R17 的证据审计已通过，但相同 seed 的 \`W_dec\` 差异仍超冻结容差；R20 的目录审计已通过，但未生成新的最终 PASS。原 round01 的 FAIL 和 INCOMPLETE 保留。
- 汇总有 **18/20 道题在两轮中至少获得过一次独立 PASS 记录**。这不代表 20 题在一个完整轮次全部通过，也不代表任意环境或更大规模任务的长期成功率。

### Harness 与产品开发边界

与考试工具直接有关的修复已完成：真实非空目录交付、路径/符号链接保护、全题资源盘点、来源模块与设备探测、命令失败完整记录、评分证据归档、多原因诊断。R17 同 seed CUDA 训练行为、Benchmark 的 \`pad_token_id\` / device 传递，以及 R20 的新最终复核，属于**后续产品实现或实际考试验收**，由独立工作安排处理，不能通过修改历史分数或放松冻结容差完成。

R08 的独立探测记录中出现 CPU float16 logits，而原题 Factory 要求 CPU float32；该事实提示今后必要时从实际 Factory/Reviewer 脚本核对 dtype。R17 的已归档比较证明同 seed 权重不一致；仅凭该比较不能把历史差异唯一归因于 CUDA TopK。5 个历史 Reviewer 目录有脚本指纹漂移，另有对应的 \`b\` 版本归档；仍需保留溯源差异记录。

## 旧 Harness 继承校验

阶段整合前，远程仓库比对显示 \`harness/tests/wp18\`、\`wp19\`、\`wp20\` 与先前 \`main\` 的 Git 子树 SHA 完全一致；\`harness/core\`、\`code\`、\`knowledge\`、\`environment\`、\`visualization\` 的子树也一致。通用 Contract catalog 与 WP18–WP20 的 reference/verifier 文件保持原有 blob SHA。新增 WP21 不得替换或降级历史 Contract 验证语义。

本轮本地整合核对共执行 **233 项定向断言通过**，另有 **28 项耗时/整树复制的验证未在本次全部复跑**（WP18/19/20 的部分大规模 Reference 检查与 WP21 的部分长耗时 Reviewer 夹具）。WP21.2 修复 PR 的开发验收记录为 **108/108**；该数字是当时分别执行的测试报告，而非本次对最终 GitHub commit 的完整 CI。初次把 ZIP 当作仓库测试时遇到 \`.git\` 缺失，改为有本地 Git 元数据的快照后旧权限门禁验证通过。真实 GPU 考试结果以各轮 Reviewer 记录为准。

本次结项只归并 Harness 工具、规则入口与报告档案，不合并各考生产品 diff，不改变正式 Atlas 树、网站、release 和产品 Runtime。原始 GPU 权重、未经人工审核的 Reviewer ZIP、敏感路径与大型张量仍保存在可信本地。

## 下一阶段如何开始

新的 Harness 任务从最新 \`main\` 读取 [统一 Tests 入口](../../tests/README.md)、[通用 Contract Registry](../../contracts/README.md)、相关 Work Plan 及对应版本的工具。运行新一轮真实考试时生成**新的 run_id、工作树和 Reviewer**；先验证资源和冻结身份，再判断行为结果。改产品代码的任务单独走研发 PR，Harness 治理修改需要明确维护者审阅。
