# H01 点名文件摘要

CASE_ID: H01  
run_id: `20261029-wp18-02`  
base_sha: `95e42deb2994b9a08e56ed14417ac3eb12ae3cbb`  
head_sha: 交付提交完成前不可用，不在候选 JSON 里自引用。

## 合同

解析结果没有 `SELECTION_GAP`。本题使用 Always-on、`CT-ERROR`，以及 `CT-EVIDENCE`、`CT-DETERMINISM`、`CT-INTERFACE`。`task_signals` 为 `source_reference` 与 `file_evidence`。

## 实现与自测

`index_named_files` 只摘要调用方点名的相对路径。空列表得到空列表。非列表、重复路径、缺失文件、`../` 和指向根外的软链接都会拒绝。未点名文件不会进入结果。

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-02/H01/test_implementation.py`  
退出码 0，5 passed，0 failed。这是本地自测，不是独立验收。

## 待人审

候选节点全部 `unchecked`。`expected_code_diff.json` 只是对 AST 的预测。Observed 需要验收者另行用 `harness/code/build.py --include-harness` 重建。
