# 四棵树与术语字典的共同准则

真正的“权威来源”有四种，必须彼此分离：

| 结构 | 回答的问题 | 数据来源 | 默认编辑方 |
| --- | --- | --- | --- |
| **Core Tree** | 项目为什么这样设计？总体目标/里程碑如何依赖？ | `harness/core/project_graph.json` | GitHub 端规划者 |
| **Knowledge Tree** | LLM SAE 特征捕获、干预、解释、评测与训练有哪些方法和验证问题？ | `doc/knowledge/context_graph.json` | GitHub 端科研上下文管理 |
| **Code Tree** | 当前仓库实际上有什么类、方法、模块关系？ | AST 生成 `doc/code/code_tree.json` | 自动化工具 |
| **Environment Tree** | 当前任务在哪台机器、哪些资源上运行？ | 探测器 / 环境快照 | 本地机器 |
| **Glossary** | Knowledge Tree 的术语统一是什么意思？ | `doc/knowledge/glossary.json` | GitHub 端科研上下文管理 |

## Core 与 Knowledge 不可混淆

Core 保留项目结构管理语义，包括原来的 `q.research`、`g.code_tree` 等 ID。Knowledge 为 CUB 分类建立 `q.cub.*`、`s.cub.*`、`g.cub.*`，并为 SAE 方法建立 `q.sae.*`、`s.sae.*`、`g.sae.*`。它们**不是同一棵树**，不应通过重用标题或复制说明互相替代。

两棵权威图只使用 `refines / addresses / decomposes` 三种父子关系。Atlas 导出额外的展示用导航分组，用于把直接子节点数量约束在十个以内；它们保留独立 ID，不改变原始研究关系。父节点进度由自己的实现与实验证据独立审计，不由子节点自动上推；不存在可选逻辑门或显式执行依赖。子节点按 JSON `edges` 中出现的先后，从上到下展示和执行。一个节点只能有一个父节点，避免交叉关系干扰树结构。

## Knowledge 的专有名词统一定义

每一个 Knowledge node 列出 `term_ids`；每个 `term.<id>` 在 `doc/knowledge/glossary.json` 只出现一次，并定义：中文/英文名、别名、明确解释、为何影响研究、依据文件、基础解释文档、关联 Knowledge 节点。两向关联必须一致，CI 校验失败就不能发布。

通用工程指标与科学术语不能以任意模型输出自动重写为新定义。知识图与术语图更新时，要一并更新 `doc/knowledge/foundations/` 和 Code Tree 的研究 ID 映射。

## Code → Core 与 Knowledge 分别映射

`doc/code/code_annotations.json` 可以同时保存 `core_ids`（管理设计依据）与 `context_ids`（科研目标）。两类引用单独校验，AST 提取出的实际类/方法关系不靠 AI 幻觉补全。Code 规划变化与 AST 实测变化必须分别报告。

## Environment 与复现

机器环境仅记录实际采样的可证实事实，未观测字段为 `unknown`；公开 `example.json` 是 `synthetic: true`。不能把硬件示例推断为研究结论，更不能上传用户的私有凭据。

实验产物由 `saebase/runtime/` 管理，Knowledge 负责解释参数、评测含义和成功条件。技术实现存在与模型独立测试达标必须严格分开。SAE 方法的绿色状态只记录有来源的操作接口；研究问题的科学状态由独立实验和评审确认。

## 叙述语句

`问题：…`、`方案：…`、`目标：…`、`证据：<可定位路径或实验运行>`、`现状：…`、`下一步：…`。状态仅为 `not_implemented`（红色）、`partial`（黄色）、`implemented`（绿色）。每个节点独立维护已审阅的状态；子节点 partial、not_implemented 或新的空提案均不会自动改写上级状态。需要查看综合进度时可以另行计算汇总，但不修改权威 JSON。科研模型未真实验证时，不得以代码已实现代替准确率已达标。
