# H02 代码树比较

CASE_ID: H02  
run_id: `20261029-wp18-02`  
base_sha: `95e42deb2994b9a08e56ed14417ac3eb12ae3cbb`  
head_sha: 交付提交完成前不可用。

## 合同

没有 `SELECTION_GAP`。额外合同是 `CT-STRUCT`、`CT-AST`、`CT-DETERMINISM`、`CT-INTERFACE`。`task_signals` 为 `tree_nodes`、`parent_edges`、`source_ast`。

## 实现与自测

`compare_code_trees` 只比较传入的两份 synthetic 树。重排不产生差异。签名变化和丢边按 node_id 报告。重复 ID、未知端点、非法关系、contains 环，以及 `project` 不是 SAEBase 的节点都会拒绝，不会补出一个新节点。函数不读取正式代码树快照。

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-02/H02/test_implementation.py`  
退出码 0，4 passed，0 failed。这是本地自测，不是独立验收。

## 待人审

Expected 与 Observed 必须分开。请用独立 AST 核对源码行号和签名，不要把本报告里的预测当成已经观察到的树差异。
