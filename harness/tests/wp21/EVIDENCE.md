# WP21 证据字段与工具范围

WP21.1 对 `command_refs` 额外要求带可信 `--control` 的录制器输出，且列举全部 `commands/NNNN.json`，受题卡累计时间预算约束。`required_artifacts` 必须完整存在；具体方法见 [Exam Ready 手册](EXAM_READY.md)。

所有路径均相对于当前候选工作树；命令日志路径相对于对应的 `commands/0001.json`。reviewer 从外置控制目录核验冻结的源文件清单。下表是 `evidence.json` 的必需内容，未执行时允许模板中的 null；真实证据审核时必须填写实际值。

| 字段 | 要求 |
| --- | --- |
| `schema` | `saebase.wp21.evidence.v1` |
| `case_id`、`run_id` | 与调度记录一致 |
| `status` | 考生可交 `NOT_RUN`、`SELF_TESTED`、`BLOCKED`/`ENV_BLOCKED`；最终独立判定另存 |
| `evidence_kind` | 正式提交为 `real_checkpoint`；CPU toy 检查只能标 `synthetic` |
| `base_commit` | 冻结基线 SHA；它不替代本地候选源码 hash |
| `source_snapshot_sha256` | 可信 `collect` 返回值，绑定初始化材料、实际代码与固定规范 |
| `input_identity` | 至少含实际输入 `sha256`、数量、sample/group IDs、split 与输入路径；不能填“默认数据” |
| `model_identity` | 来源、40/64 位不可变 `revision`、`files_sha256`、实际模型 key、dtype、device；`files_sha256` 引用外置模型资源锁中声明的对应文件集合 digest，由 reviewer核对 |
| `sae_identity` | 实际 SAE 来源、版本、权重 hash、hook、层和维度；R17还含本次训练 checkpoint。R08/R09 model-only 时为 null |
| `environment` | 实际 Python/依赖版本、设备及 `saebase` 实际导入源码位置；非推测的环境值 |
| `resolved_config_path` | 本次实际 resolved config 的 JSON 文件，建议放本题 run 的 `metadata/resolved_config.json` |
| `runtime_config_sha256` | 原 `metadata/run.json` 的 YAML 文本 hash，API-only 无此字段时明确 null；不重命名成 canonical hash |
| `canonical_config_sha256` | 对上述 JSON 内容以 `ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False` 编码 UTF-8 后 SHA256 |
| `executed_result_paths` | 至少一个真正执行结果 JSON 的路径，必须在 `experiment/` 且被完整产物清单覆盖；不能指向配置打印或 dry-run |
| `command_refs` | 有序去重，例如 `['commands/0001.json','commands/0002.json']`；包含失败命令，和 receipt 的 argv/exit_code一一对齐 |
| `unresolved` | 实际未解决或未执行项，不删除失败历史 |

输入文本可以是本包原创的受控测试文本；真实 checkpoint 与 synthetic 输入是两个独立事实。报告可用 `input_identity.kind=controlled_synthetic_prompts`，同时使用 `evidence_kind=real_checkpoint`。

`contract_receipt.json` 使用 WP21 专有 schema，保留 `registry_version=1.1.0`、逐题固定 `contract_ids`、实际 `context_paths`、所有观察到的产品修改 `edited_paths`、与 command_refs 同序的 `commands=[{'argv':[...],'exit_code':0}]` 和 unresolved。工具生成的证据文件不计为产品源码编辑。

`source_manifest.json` 和 `artifacts_manifest.json` 由可信收集器写入。后者穷举当前候选 `experiment/` 下的所有文件，相对路径、字节数及 SHA256；它本身在证据目录，不进入自身递归哈希。自写清单可作为题目产物，但不能覆盖独立收集器的观测。

`expected_code_diff.json` 记录执行前的增删改预期。`review_candidate.json` 只提交 unchecked 候选，新增 node 的中文/英文标题、不同于标题的说明、Context 词条、源代码引用必须齐全；当题目无源码变更时可以明确空变化并说明理由。对实际 AST 的比较由独立 reviewer执行。

## 状态含义

| 工具结果 | 能说明什么 |
| --- | --- |
| `SPEC_VALID` | 20题、绑定、路径、fixture哈希可解析 |
| `PREREQUISITES_PRESENT_UNVERIFIED` | 已提供依赖和目录；尚需真实加载探测 |
| `COLLECTED_NOT_GRADED` | 已收集实际源码差异和产物文件 |
| `EVIDENCE_READY_FOR_REPLAY` | 结构、状态和哈希一致，具备进一步独立复跑的材料 |
| 独立 `PASS` | reviewer实际重跑题卡的正反例、数值复算和兼容检查，全部硬条件满足 |

可伪造的状态字段和 hashes 不能证明模型确实执行。自动工具不会判断论文结论、模型解释有效性或任意源码语义。真实行为、实际导入路径、资源锁、数值、负例和候选代码必须由独立执行核验。
