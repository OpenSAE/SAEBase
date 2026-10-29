# H05 DFS 审阅队列

CASE_ID: H05  
run_id: `20261029-wp18-02`  
base_sha: `95e42deb2994b9a08e56ed14417ac3eb12ae3cbb`  
head_sha: 交付提交完成前不可用。

## 合同

没有 `SELECTION_GAP`。额外合同是 `CT-STRUCT`、`CT-REVIEW`、`CT-INTERFACE`。`task_signals` 为 `tree_nodes`、`parent_edges`、`review_state`。

## 实现与自测

`queue_nodes` 按兄弟 `order` 做深度优先，支持 all、tree、parent、node。未知节点不会被造出来。跨树父边和环会拒绝。草稿带修订号；过期修订失败且不改原状态。子节点的 deny 不改变父节点。`undo_draft` 只回到 unchecked。`refuse_authority_write` 固定失败。导出的 `formal_approval` 为 false，没有批准端点。

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-02/H05/test_implementation.py`  
退出码 0，5 passed，0 failed。这是本地自测，不是独立验收。

## 待人审

草稿不是 Approve。不需要修改 `agent_flow.py`，因此本题没有 BLOCKED。请确认候选节点仍是 unchecked。
