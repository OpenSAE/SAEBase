# Code · 实际接口与结构可视化

`harness/code/build.py` 使用 AST 提取真实 Python 模块、类、方法和代码关系。结果保存到 `doc/code/code_tree.json`。展示层 `harness/code/presentation.py` 提供自然语言名称以及“具体数据集 / 具体模型 / 具体任务”分组。

`harness/code/interfaces.py` 针对已有代码提取函数的参数、类型、默认值、返回表达式、真实源码；类节点额外汇总初始化成员字段及**全部方法**。界面在节点卡片内展示这些属性，直接对应原始 Python 标识。

```bash
python harness/code/build.py
python harness/code/build.py --check
python harness/visualization/export.py
python harness/code/diff.py --before /tmp/code_before.json --after doc/code/code_tree.json
```

