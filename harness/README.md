# OpenResearch Harness · 按需激活

这是独立于生产模型/数据/任务的开发支持层。六个功能模块：`core/`、`code/`、`knowledge/`、`experiments/`、`environment/`、`transfer/`；`visualization/` 是公共 UI，不计为科研功能模块。默认 Agent 不主动读完整 Harness，只有用户或 Work Order 明确激活的模块参与任务。

| 模块 | 职责 | 权威数据 |
| --- | --- | --- |
| [core/](core/) | 全项目架构、研发管理、任务优先级、模块边界和设计理念 | `harness/core/project_graph.json` |
| [code/](code/) | AST 源码树、类/函数/方法、依赖与结构变更 Diff | `doc/code/code_tree.json`（派生快照） |
| [knowledge/](knowledge/) | 具体科研问题的知识树、术语校验、节点选择和状态提案 | `doc/knowledge/context_graph.json` + `glossary.json` |
| [experiments/](experiments/) | 结果目录/子实验/元数据/复现检查 | `dlbase/experiment_paths.py` |
| [environment/](environment/) | CPU/GPU/RAM 与运行环境观测 | 本地探测器产生的真实快照 |
| [transfer/](transfer/) | Backblaze B2 / rclone 文件复制与校验 | 明确的来源、目标与本地凭据 |
| [visualization/](visualization/) | 统一 Atlas：Core、Knowledge、Code、Environment、Context | 上述权威数据导出的只读静态视图 |
| [github_work_plan/](github_work_plan/) | 明确要求时读取的临时参考资料 | 当前任务清单 |

### 暂缓工作计划

- [OpenResearch 实机功能测试与科研验收完整计划](github_work_plan/OPENRESEARCH-FUNCTIONAL-ACCEPTANCE-DEFERRED.md) — **DEFERRED / 暂不执行**。当前只推进 Harness 测试和验收；Harness 完整验收之后，须经负责人重新确认，才考虑启动具体功能测试。该文件不是激活的 Work Order，也不授权 GPU 运算、模型下载或功能修改。

**两棵不同的语义树**：Core 管“这个项目怎样设计和组织”；Knowledge 管“如何训练与评价 CUB-200 的 ResNet 分类模型”。当前只展示已经存在的 CUB 分类组件，不额外加入尚未开发的能力；不能从源码名推断未经证实的科研结论。

本地刷新：

```bash
python harness/core/validate.py
python harness/knowledge/validate.py
python harness/code/build.py
python harness/visualization/export.py
python harness/visualization/build_pages.py
python -m http.server 8765
```

浏览 `http://localhost:8765/harness/visualization/index.html`。GitHub Pages 公共版本是**静态快照**，更改私有仓库后需单独更新公开镜像。

`core/GITHUB_ONLY.md` 为 GitHub 侧规划说明，不应被本地 Agent 直接读取；忽略规则并非操作系统级隔离。

## 现有能力的公开投影

内部 Core/Knowledge 权威数据仍保留历史未完成记录，公开 Atlas 只导出绿色 `implemented` 节点。未完成的上级目标被隐藏时，已实现的子节点连接到最近的绿色祖先或纯分组根。这种分组不是新增功能。Code Tree 的原始 AST 保留完整，公共展示仅取主要模块、类及公开方法。Environment 不使用绿色过滤，可保留观测或未知状态。

## Agent 执行任务的流程

外部 github_work_plan 先经 [Harness Core 审核流程](core/AGENT_FLOW.md) 审计，并在 [Atlas Knowledge 审阅模式](visualization/knowledge.html?review=OR-Knowledge-Demo-20261026) 中预览 Knowledge、Code、Environment、Context 的红色变化。代码审阅后黄色，实验结果另行审计后由人类绿/黄/红/删除逐节点决定。普通 Agent 不编辑 Core/规则。
