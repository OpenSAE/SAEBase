# H03 术语引用审计

CASE_ID: H03  
run_id: `20261029-wp18-02`  
base_sha: `95e42deb2994b9a08e56ed14417ac3eb12ae3cbb`  
head_sha: 交付提交完成前不可用。

## 合同

没有 `SELECTION_GAP`。额外合同是 `CT-EVIDENCE`、`CT-CONTEXT`、`CT-INTERFACE`。`task_signals` 为 `source_reference` 与 `glossary_terms`。

## 实现与自测

`audit_term_links` 检查双向 `term_ids` / `related_node_ids`。来源要能在给定根目录里读到对应行，才标成 resolved。父目录跳转被拒绝；文件不存在标成 unresolved，不当成已核实。同义词冲突只产生未执行建议，不写 Glossary，也不把节点标成 implemented。

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-02/H03/test_implementation.py`  
退出码 0，5 passed，0 failed。这是本地自测，不是独立验收。

## 待人审

这些术语 ID 只存在于本题 fixture，不是 Glossary 里的新权威术语。`unresolved` 保持为提案。
