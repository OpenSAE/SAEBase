# 源码结构与功能说明

## 目的和主要功能

介绍源码树、源码说明和科研概念的关联，帮助维护者定位实现及其用途。

## 适用范围

用于读取或维护 `doc/code/` 的源码结构与标注资料。实际修改依据当前源码和项目工作模式。

`code_tree.json` 保存源码语法结构，`code_annotations.json` 保存源码用途和相关知识标识。`context_ids` 关联项目知识，`core_ids` 关联中央管理，`project_architecture_ids` 关联项目自身架构。说明以实际源码和已知依据为准；含义待查时保留具体位置和后续定位线索。

维护功能说明时，直接写清该实现处理什么、完成什么、产生什么以及谁使用结果。参数和实现细节用于需要查看这些信息的接口资料，整体说明先解释职责和协作。

[当前架构](development/current-architecture.md)说明各组件职责，[开发文档](development/index.md)提供具体使用入口。需要中央源码图或 Atlas 时，通过 `.researchharness/central.json` 定位 ResearchHarness，按其源码与可视化说明处理。
