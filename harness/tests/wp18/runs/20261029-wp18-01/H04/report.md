# H04 合成环境与实测隔离

CASE_ID: H04  
run_id: `20261029-wp18-01`  
base_sha: `8018a2cd7dc0a8a7035e860fcaaaf17a18f9f07f`（GitHub 快照。本工作区没有 `.git`。）  
head_sha: 未提交。

## 做了什么

`normalize_environment` 只接受 `synthetic: true` 的记录。空字段和出现 `nvidia-smi` 字样的值写成 `unknown`，不用它们推断“无 GPU”。`token`、`secret` 和绝对路径只留下字段名。返回值里 `observed` 为 false，`measurement` 为 `not_observed`。没有调用 GPU 探测。

候选里另有一个 `view: environment` 的节点，标明这是合成概述，不是权威 Environment 树记录。

## 路径

- 实现：`harness/tests/wp18/runs/20261029-wp18-01/H04/implementation.py`
- 自测：`test_implementation.py`
- 合成数据：`fixtures/device.json`
- 候选：`review_candidate.json`（4 个 code 节点加 1 个 environment 候选，全部 unchecked）
- 预测：`expected_code_diff.json` 只含 Python AST，不含那个 environment 候选 ID

## 自测

`python3 -m pytest -p no:cacheprovider -q harness/tests/wp18/runs/20261029-wp18-01/H04/test_implementation.py`  
退出码 0，5 passed，0 failed。

## 没有做

没有运行 `nvidia-smi`，没有记录真实用户路径或密钥，没有把合成值写成机器实测或研究任务成功。

## 请人审

看 environment 候选是否被误当成已观测硬件。代码节点仍须与 `normalize_environment`、`_secret`、`_public_value` 和 `EnvironmentRecordError` 的源码行号一致。
