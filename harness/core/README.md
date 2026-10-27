# Core · 项目管理与总体设计

**Core Tree** 保留过去的 30 个项目级目标及稳定 ID，并移动到 [project_graph.json](project_graph.json)。现已去掉条件逻辑和执行依赖；子节点按从上到下的顺序排列，所有直接子目标完成后父节点才算完成。它只描述 OpenResearch 的架构、基础设施、实验工程和研发决策。

```bash
python harness/core/validate.py
python harness/core/select.py q.research s.explainability g.code_tree
python harness/core/manage.py propose --node g.code_tree --status partial --reason "有实际证据后提交改变提案" --evidence harness/code/build.py
```

**GitHub 端**维护项目方向、分解和状态；Agent 一般只能按 Work Order 读取少量 `core_ids`，提交含证据的 proposal，不能默认覆写官方管理状态。

- [TREE_CONTRACTS.md](TREE_CONTRACTS.md) 是 Core / Knowledge / Code / Environment 的来源权威与引用协议。
- [BOUNDARIES.md](BOUNDARIES.md) 约束各模块读写职责和新模块标准。
- [GITHUB_ONLY.md](GITHUB_ONLY.md) 面向上层 GitHub 规划；禁止 Cursor Agent 日常浏览（不是权限沙箱）。
- [idea/](idea/) 是尚未实施的建议。

[Core Tree 在线查看](https://atheavenma.github.io/opensae/openresearch/atlas/core.html)。

## Agent 执行任务的流程

具体的审计、预览、授权、源码审阅、结果观察与四种人类最终决定见 [AGENT_FLOW.md](AGENT_FLOW.md)。该执行流程屏蔽 Core Tree；新增 Core 节点只由治理方维护，当前标记 partial 直到实机验收。
