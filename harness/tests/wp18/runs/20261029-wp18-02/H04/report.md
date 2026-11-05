# H04 合成环境概述

CASE_ID: H04  
run_id: `20261029-wp18-02`  
base_sha: `95e42deb2994b9a08e56ed14417ac3eb12ae3cbb`  
head_sha: 交付提交完成前不可用。

## 合同

没有 `SELECTION_GAP`。额外合同是 `CT-PRIVACY`、`CT-ENV`、`CT-INTERFACE`。`task_signals` 为 `sensitive_output`、`credentials`、`environment_observation`。

## 实现与自测

`normalize_environment` 只接受 `synthetic: true`。字典和列表最多展开四层；布尔值、过深嵌套等无法完整检查的值会整份拒绝，异常文本不带出原密钥。空字段和 `nvidia-smi` 字样写成 unknown，不用它们推断没有 GPU。嵌套的 token、secret 和绝对路径只留下 `[redacted]`。

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-02/H04/test_implementation.py`  
退出码 0，4 passed，0 failed。这是本地自测，不是独立验收。

## 待人审

候选里的 environment 节点标明 synthetic，不是权威 Environment 树里的实测记录。`observed` 在结果中固定为 false。
