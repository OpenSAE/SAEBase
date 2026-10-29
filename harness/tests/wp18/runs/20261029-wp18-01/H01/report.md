# H01 文件证据索引

CASE_ID: H01  
run_id: `20261029-wp18-01`  
base_sha: `8018a2cd7dc0a8a7035e860fcaaaf17a18f9f07f`（同步到的 GitHub 快照。本工作区没有 `.git`，不能做本地 `rev-parse`。）  
head_sha: 未提交。候选 JSON 只记录 `implementation.py` 的 `source_content_sha256`，不写自引用提交号。

## 做了什么

`index_files` 对调用方点名的相对路径输出按路径排序的 `path / sha256 / size`。拒绝 `../`、绝对路径、重复路径、缺失文件，以及指向根外的软链接。未点名的文件不会被读取。

## 路径

- 实现：`harness/tests/wp18/runs/20261029-wp18-01/H01/implementation.py`
- 自测：`test_implementation.py`
- 合成数据：`fixtures/alpha.txt`、`fixtures/beta.txt`、`fixtures/manifest.json`
- 候选：`review_candidate.json`（4 个 code 节点，`provisional: true`，`status: unchecked`）
- 预测：`expected_code_diff.json`（含本目录两份 Python 的模块、类、函数和边；不是 Observed）

## 自测

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-01/H01/test_implementation.py`  
退出码 0，7 passed，0 failed。

## 没有做

没有扫描仓库、没有读取凭据、没有写 `acceptance/`、权威树、`.cursor/`、`harness/core/` 或 `saebase/runtime/`。没有宣布独立验收通过。Context 术语未创建，见候选里的 `unresolved`。

## 请人审

核对 `index_files`、`_require_relative`、`_resolve_inside` 和 `EvidenceIndexError` 的行号、签名是否与源码一致，并确认 Expected 不能代替验收者重建的 AST。
