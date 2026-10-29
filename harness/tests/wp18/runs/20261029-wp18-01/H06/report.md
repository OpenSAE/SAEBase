# H06 双语覆盖审计

CASE_ID: H06  
run_id: `20261029-wp18-01`  
base_sha: `8018a2cd7dc0a8a7035e860fcaaaf17a18f9f07f`（GitHub 快照。本工作区没有 `.git`。）  
head_sha: 未提交。

## 做了什么

`audit_overlay` 按稳定 ID 检查小型 zh/en 记录。它报告缺英文、标题与摘要在去掉标点后相同、`English translation unavailable`、U+FFFD、项目不是 SAEBase、本地化签名不同于源码签名，以及签名里的下划线符号只在英文散文中被拆开。`preserved` 原样保留 id、signature 和 path。`fixed` 始终是空列表，因为函数不修改翻译表。

## 路径

- 实现：`harness/tests/wp18/runs/20261029-wp18-01/H06/implementation.py`
- 自测：`test_implementation.py`
- 合成数据：`fixtures/overlay.json`
- 候选：`review_candidate.json`（7 个 code 节点，unchecked）
- 预测：`expected_code_diff.json`

## 自测

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-01/H06/test_implementation.py`  
退出码 0，3 passed，0 failed。

## 没有做

没有编辑正式中英翻译表、Atlas 页面或权威 Code/Context 树。检测到的问题没有被标成已修复。

## 请人审

用 `fixtures/overlay.json` 对照 `detected` 与 `fixed`。候选节点是审计器源码，不是已经修正过的双语树。
