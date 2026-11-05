# H06 双语行审计

CASE_ID: H06  
run_id: `20261029-wp18-02`  
base_sha: `95e42deb2994b9a08e56ed14417ac3eb12ae3cbb`  
head_sha: 交付提交完成前不可用。

## 合同

没有 `SELECTION_GAP`。额外合同是 `CT-LOCALE`、`CT-AST`、`CT-INTERFACE`。`task_signals` 为 `bilingual_fields` 与 `source_ast`。

## 实现与自测

`review_locale_rows` 同时检查中文和英文。缺任一种语言、标题与摘要在去掉标点后相同、`English translation unavailable`、U+FFFD、项目不是 SAEBase、本地化签名不同于源码签名，以及下划线符号被拆进英文散文，都会进入 `detected`。`preserved` 原样保留 id、signature 和 path。`fixed` 始终为空，函数不补译文，也不改翻译表。

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-02/H06/test_implementation.py`  
退出码 0，3 passed，0 failed。这是本地自测，不是独立验收。

## 待人审

候选节点是审计器源码的中英标题和摘要，不是已经修好的双语树。请对照 `fixtures/rows.json` 区分检测到的问题和未修复的问题。
