# Code Architecture

`code_tree.json` 是 AST 自动生成的数据；生成器在 `harness/code/build.py`，结构 Diff 在 `harness/code/diff.py`，不是手工绘制的模型能力总结。

`code_annotations.json` 使用 `context_ids` 引用具体科研 Knowledge（`doc/knowledge/context_graph.json`），使用 `core_ids` 引用项目设计 Core（`harness/core/project_graph.json`）。二者不得混合使用；没有经证实的描述保留 needs_review。

浏览 [Code Tree](https://atheavenma.github.io/opensae/openresearch/atlas/code.html) 时，每个 Class / Method 节点可查看对应源码及来自两棵语义图的不同依据。
