# H03 术语引用一致性

CASE_ID: H03  
run_id: `20261029-wp18-01`  
base_sha: `8018a2cd7dc0a8a7035e860fcaaaf17a18f9f07f`（GitHub 快照。本工作区没有 `.git`。）  
head_sha: 未提交。

## 做了什么

`check_context_links` 在带 `synthetic: true` 的小型节点和术语包上检查 `term_ids`、`related_node_ids` 和形如 `path:line` 的 `source_evidence`。它报告缺失术语、缺失节点、反向不一致、重复 ID、无法定位的证据，以及两个术语共用同一别名。建议的 `applied` 固定为 `false`。`glossary_written` 为 false，`implemented_nodes` 为空。同义词不会被合并成新术语。

## 路径

- 实现：`harness/tests/wp18/runs/20261029-wp18-01/H03/implementation.py`
- 自测：`test_implementation.py`
- 合成数据：`fixtures/consistent.json`、`broken.json`、`aliases.json`
- 候选：`review_candidate.json`（7 个 code 节点，unchecked）
- 预测：`expected_code_diff.json`

## 自测

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-01/H03/test_implementation.py`  
退出码 0，4 passed，0 failed。

## 没有做

没有修改 `doc/knowledge/glossary.json` 或 Knowledge 权威图，没有从函数名发明科学术语。

## 请人审

这些诊断只针对测验 fixture。`unresolved` 写明别名冲突不是新的 Glossary 术语。请核对七个源码节点的行号和签名，不要把建议当成已经落地的术语修改。
