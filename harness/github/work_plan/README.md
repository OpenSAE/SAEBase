# SAEBase Work Plan · 当前执行与审批入口

Work Plan 是交给 Local Agent 的一份独立、已准备好的执行计划；规划完成不代表获得执行或发布权限。除人类批准的明确目的外，不得引用其他 Work Plan 作为任务依赖。

## 必须遵守的现行规范

- [编写标准](WRITING_STANDARD.md) · [英文](WRITING_STANDARD.en.md)
- [独立计划模板](WORKPLAN_TEMPLATE.md) · [人类确认的需求](from_human/README.md)
- [交付与验证标准](DELIVERABLE_STANDARD.md)
- [计划生命周期验证](lifecycle.py) · [源码影响与审阅检查](review_gate.py)
- [任务调度接口](dispatch.py)

先明确用户已确认的目标、现有接口与文件归属、可执行命令、环境依赖和独立验收。每项源码修改必须得到精确范围的 Human Check 或其他现有授权；Atlas 长期节点仅在职责和性质确实发生变化时单独申请修改。

只有明确的人工批准才能在不同 Work Plan 之间建立必要的引用关系。未批准的执行步骤必须处于 Draft / Blocked，而不能交给 Local Agent 自行决定。

## 当前计划

[组件检查与审计任务](0.6.5_wp023_qa_precheck.md) — 具体执行状态与限制以该计划声明及本次人类授权为准。计划内尚未完成的科学和硬件验证不构成 PASS。
