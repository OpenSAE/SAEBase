---
name: researchharness
description: 在常规模式接入中央 ResearchHarness，或按用户要求切换 SAEBase 开发模式时使用。
---

# 中央 ResearchHarness 接入

## 目的和主要功能

使用本项目显式绑定的中央管理入口，并按用户指令切换开发模式。

## 适用范围

适用于常规模式下的管理操作、用户指定的中央管理任务和模式切换。项目工作模式由根 AGENTS.md 统一说明。

## 接入与切换

- 读取 `.researchharness/central.json` 的中央路径及 `.researchharness/project.yaml` 的科研输入声明。中央项目标识为 `saebase`。
- 查询使用 `rh --harness-root <中央目录> --project saebase status`。项目注册与归属使用中央 `src/harness/core/PROJECT.md`；源码审阅使用 `src/harness/code/CODE.md`；正式计划使用 `src/harness/core/work_plan/workplan.md`，按当前任务选择阅读。
- 常规模式的源码审阅参数 `human_check_required` 默认 `true`，按用户明确指令和范围调整；生产操作采用当前中央执行授权。普通编辑与软件检查按现行规则处理。
- 模式切换使用 `rh --harness-root <中央目录> --project saebase mode set intensive` 或 `mode set regular`，查询使用 `mode status`。
- 使用中央查询或检查结果时，定位实际对象、具体条件与可继续的动作；维护接入功能时，在制定接口阶段安排这些结果如何被项目读取和使用，已有原因与来源随调用保留。
- 中央源码和配置的修改依据相应维护授权；SAEBase 科研程序独立安装运行。历史批准作为历史依据，当前任务采用现行授权。
