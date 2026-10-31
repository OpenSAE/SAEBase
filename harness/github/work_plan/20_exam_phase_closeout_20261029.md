# Work Plan 20 · Harness 考试探索阶段结项

记录日期：2026-10-29  
结项性质：**隔离 CPU 行为考试探索完成，历史成绩归档**。

## 1. 本阶段实际验证了什么

本阶段以 SAEBase 仓库内的 Harness 为考试环境，验证 Agent 能否按照预先声明的 Work Plan、Universal Contract 与冻结行为参考完成二十个独立的合成任务（X01–X20）。每题包含两个公开合法场景和两个公开异常场景，总计 80 个公开检查。

**这里的“通过”只指 Harness 中定义的 CPU 行为考试**。任务会涉及 AST、数据、SAE 数值运算、归档及临时 Git 工作区，但多数执行对象是为考试创建的合成输入与隔离实现，而不是对真实 SAEBase 生产工作流进行调用与验收。因此本阶段完成的是 Harness 考试探索，不是产品级集成认证。

## 2. 归档的版本与证据

| 项目 | 记录 |
| --- | --- |
| 考试设计 | [PR #94](https://github.com/OpenSAE/SAEBase/pull/94)，已合并 |
| 可信行为参考与公开评分器 | [PR #95](https://github.com/OpenSAE/SAEBase/pull/95)，已合并 |
| Agent 考试提交 | [PR #96](https://github.com/OpenSAE/SAEBase/pull/96)，已合并至 main 归档 |
| Run ID | `20261029-wp20-01` |
| 考试 Base SHA | `df9b8bb68789690e1634089a22c1513f06ac8619` |
| Agent 提交 SHA | `f3d1b645c46451799435e7cf145bab56d0e7ba40` |
| 归档合并 SHA | `b04a12c14c7e737ed0624b888ec68f1d16e2e825` |
| 新增考试文件 | 260 |
| Agent 自检报告 | 20/20 题、80/80 公开场景 PASS；自有 pytest 共 163 项 |
| GitHub CI | [CI 成功](https://github.com/OpenSAE/SAEBase/actions/runs/40956281851) |
| GitHub Trusted Scope | [允许写入范围检查成功](https://github.com/OpenSAE/SAEBase/actions/runs/40956281358) |

**证据等级**：上述 80/80 与 163 项记录来自考生提交的自测材料。GitHub CI 和 Trusted Scope 结果可独立核实，但普通 CI 没有在可信 OS 隔离环境中逐题重新运行所有考生代码。归档本次提交表示保存并认可其阶段探索价值，不代表人工审核了所有原始输出，也不代表正式部署批准。

## 3. 本阶段结论

1. 二十题的规格、Reference、统一检查入口与错误实现校准已经可用于受控合成考试。
2. Agent 完成二十题交付，公开自检均为 PASS，正常仓库 CI 与提交路径权限审计成功。作为 Harness 约束与考试工作流的阶段实验，这一结果达到归档条件。
3. 目前考试的被测对象主要位于 Harness 自己的合成运行环境。部分 Work Plan 的较大目标被缩减成局部函数行为。核心限制是**测试环境与真实 SAEBase 运行环境的距离**，不是本次 Agent 必须针对旧场景继续刷题。
4. [独立源码审阅记录](https://github.com/OpenSAE/SAEBase/pull/96#pullrequestreview-5705463975)发现候选 Code Tree 节点的双语描述重复、X01 manifest 自引用哈希不一致等交付证据缺陷。它们作为后续改善 Harness 交付核验的历史发现保留，不修改已冻结的考试标准，也不提升为生产审批通过。

## 4. 阶段性处理决定

- **WP20 考试探索完成并归档**。合并 PR #96 是保存历史考试记录，不是将题目解决方案作为 SAEBase 正式产品功能发布。
- 保留 `wp20_cases.json`、公开场景、参考实现和现有评分器的历史定义，避免在结项时重写已执行试卷。
- 保留 Agent 的原始 `selfcheck`、`run_trace`、`contract_receipt` 和候选审阅材料，后续如有修订应进入新的、明确标记的版本，而不是静默覆写。
- 历史 Reviewer 发现继续可追溯；正式 Atlas 节点审批、SAEBase 产品集成验收和真实环境运行均保持独立。

## 5. 后续研究方向，仅作边界记录

未来若继续探索 Agent 在真实 SAEBase 中的开发能力，应将工作重心转向实际仓库接口、真实任务工作流、可重启进程与真实产物，而不仅是对 Harness 内参考函数的复现。独立 Reviewer 应从真实输入、过程与产物核实效果。

这属于**下一阶段的不同研究问题**，不作为 WP20 结项的补考要求。本阶段至此结束。
