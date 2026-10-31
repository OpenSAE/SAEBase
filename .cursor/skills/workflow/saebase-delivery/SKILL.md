---
name: saebase-delivery
description: 使用 GitHub 生成的 SAEBase Work Plan 和交付指令时，自动选择安全的本地 Precheck、按阶段执行并反馈证据。
---

# SAEBase 本地交付

这是本地执行入口，规范来源为仓库的
`.github/skills/saebase-delivery/SKILL.md` 和
`harness/github/development_doc/agent-delivery.md`。
正式生产任务仍按既有 `run-work-order`、Agent Flow 和 Universal Contracts 执行。

1. 只接收用户指定的 Work Plan、Work Order、精确分支和 SHA。先确认 Git 工作区和文件范围；不得自动清理、切换覆盖未提交的文件或扩展授权。
2. 运行 `python scripts/workplan/delivery_precheck.py --mode local --profile auto --expected-branch <branch> --expected-sha <sha> --work-order <ID> --run-safe`；无 Work Order 的受限 Harness 考试遵守原有 sandbox 例外，不将缺失授权视为 PASS。
3. 预检只执行本机只读检查，环境检查必须在实际执行机器上完成。依赖解析、权重、许可证、GPU 和真实 Pilot 需按已批准的 Work Plan 另行执行。
4. 出现 FAIL/BLOCKED 就停止受影响阶段，保留原始命令、退出码和日志；不得改 Contract、指标、模型身份或 Hook 来使检查变绿。
5. 按既定任务依赖顺序执行。修改代码先记录 Expected，完成后比较 Observed AST，运行目标测试并保存真实证据。
6. 交还准确 base/head SHA、文件 diff、Contract IDs、命令/退出码、PASS/FAIL/BLOCKED/NOT_RUN、产物哈希和剩余待审批事项。本地 Agent 不自行宣布独立验收或发布。

只有在用户实际授权后才能 Push/PR、运行付费计算或发布；Precheck 与 Skill 不能替代原有治理审批。
