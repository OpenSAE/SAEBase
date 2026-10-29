# H05 人类审阅 DFS 队列

CASE_ID: H05  
run_id: `20261029-wp18-01`  
base_sha: `8018a2cd7dc0a8a7035e860fcaaaf17a18f9f07f`（GitHub 快照。本工作区没有 `.git`。）  
head_sha: 未提交。

## 做了什么

`build_queue` 在 `synthetic` 图上按兄弟节点的 `order` 做深度优先，支持 all、tree、parent、node。名称和 `wp-01` / `wp-02` 这类编号不决定顺序。草稿存在调用方传入的状态里；切换范围不会清掉已有草稿。子节点的 deny 不改变父节点。`undo_draft` 只回到 unchecked。`serialize_drafts` 的 `formal_approval` 固定为 false，`human_approval_endpoint` 为 null。`reject_formal_approval` 总是失败。

不需要改 `agent_flow.py`，因此本题没有报 BLOCKED。

## 路径

- 实现：`harness/tests/wp18/runs/20261029-wp18-01/H05/implementation.py`
- 自测：`test_implementation.py`
- 合成数据：`fixtures/graph.json`
- 候选：`review_candidate.json`（12 个 code 节点，unchecked）
- 预测：`expected_code_diff.json`

## 自测

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-01/H05/test_implementation.py`  
退出码 0，5 passed，0 failed。

## 没有做

没有调用生产审阅写入端点，没有把任何节点标成已批准，没有修改 Agent Flow。

## 请人审

草稿颜色只存在于测验状态对象里。请确认候选节点全部仍是 unchecked，并且没有人把 `draft: approve` 读成正式 Approve。
