# H02 代码树结构差异

CASE_ID: H02  
run_id: `20261029-wp18-01`  
base_sha: `8018a2cd7dc0a8a7035e860fcaaaf17a18f9f07f`（GitHub 快照。本工作区没有 `.git`。）  
head_sha: 未提交。

## 做了什么

`diff_trees` 只比较调用方传入的两份测验 JSON。按稳定 `node_id` 报告新增、删除、签名变化和边变化。纯重排不产生差异。重复 ID、未知端点和 `contains/calls/imports/inherits` 以外的关系会失败。函数不打开正式代码树快照。

## 路径

- 实现：`harness/tests/wp18/runs/20261029-wp18-01/H02/implementation.py`
- 自测：`test_implementation.py`
- 合成数据：`fixtures/before.json`、`reordered.json`、`signature.json`、`structure.json`、`duplicate_id.json`、`unknown_endpoint.json`、`illegal_relation.json`
- 候选：`review_candidate.json`（4 个 code 节点，unchecked）
- 预测：`expected_code_diff.json`

## 自测

同一命令跑了两次：

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-01/H02/test_implementation.py`

1. 退出码 1。`test_does_not_name_authority_tree` 失败：模块文档字符串写了正式快照的路径，断言要求源码中不出现该字符串。5 passed，1 failed。
2. 文档字符串改为“不打开仓库中的正式代码树快照”，并按新源码重写了候选哈希。退出码 0，6 passed，0 failed。

第一次失败留在本报告和 `selfcheck.json` 的 `exit_codes: [1, 0]` 里，没有删测试。

## 没有做

没有读取或覆盖 `doc/code/code_tree.json`，没有运行 `harness/code/build.py`，没有把 Expected 写成 Observed。

## 请人审

用独立 AST 重建核对 `diff_trees`、`_validate`、`_pack` 和 `TreeDiffError`。重排、签名变化和丢边的行为以 `fixtures/` 为准，不以来自本报告的叙述为准。
