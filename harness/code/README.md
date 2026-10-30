# Code · 实际接口与结构可视化

`harness/code/build.py` 使用 AST 提取真实 Python 模块、类、方法和代码关系。结果保存到 `doc/code/code_tree.json`。展示层 `harness/code/presentation.py` 提供自然语言名称以及“具体数据集 / 具体模型 / 具体任务”分组。

`harness/code/interfaces.py` 针对已有代码提取函数的参数、类型、默认值、返回表达式、真实源码；类节点额外汇总初始化成员字段及**全部方法**。界面在节点卡片内展示这些属性，直接对应原始 Python 标识。

```bash
python harness/code/build.py
python harness/code/build.py --check
python harness/visualization/export.py
python harness/code/diff.py --before /tmp/code_before.json --after doc/code/code_tree.json
```


## 功能代码迁移与 Code Tree

历史工作计划编号属于 `harness/github_work_plan`，不得直接作为可执行 Python 包或 Atlas 功能节点。原 `saebase/wp12/` 中仍在使用的实现已按职责迁入 `saebase/interventions/`、`saebase/circuits/`、`saebase/benchmarks/`、`saebase/sae/` 和 `saebase/release/`。Atlas 从当前 Python 源码目录重建完整功能树；研发记录保留历史名称和原始证据，不再作为现役源码接口。
