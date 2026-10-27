# Knowledge 与 Context

科研知识图保存在 `doc/knowledge/context_graph.json`，统一术语保存在 `doc/knowledge/glossary.json`，展开式解释保存在 `doc/knowledge/context_attributes.json`。

每项知识的定义采用肯定式简洁描述。具体代码概念与 Python 函数关联，包含输入参数、返回值与源码主要部分。图示和公式用于解释实际概念。

```bash
python harness/knowledge/validate.py
python harness/knowledge/context_attributes.py
python harness/knowledge/select.py q.cub.train s.cub.model
python harness/knowledge/terms.py term.resnet18 term.logits
```

[Context 浏览器](https://atheavenma.github.io/opensae/openresearch/atlas/context.html) 支持在对应节点卡片中展开文字、公式、示意图及实际代码。
