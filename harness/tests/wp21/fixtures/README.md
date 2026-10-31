# WP21 冻结输入与删除题初始状态

本目录由出题者维护，执行 agent 只读。五个删除题从相同 SAEBase 基线加本题 overlay 开始，彼此独立。这里提供的是待删除功能确实存在的初始状态，尚未替执行者完成删除。

## 公共文本

`common/prompts.jsonl` 有24条为本次考试原创的简短文本。每行字段严格为 `sample_id`、`prompt_group_id`、`split`、`prompt`。train12、validation6、test6，12个 group各有两条相邻变体，前6组train，中间3组validation，最后3组test；各 split 无交叉。`common/manifest.json` 固定字节 SHA、按 split 列出的样本 ID 与来源声明。

这是工程验收输入。训练例可为达到明确 token 预算重复 train12 中的文本，但唯一观察仍是12条。validation/test 不可用于拟合或训练；参数选择用 validation，最后选择结束后才使用 test。实际 tokenizer 长度由真实运行记录，词数不代替 token 数。

## 安装 seed

调度器在本题专属的本地 source copy 中验证 `manifest.json` 的 base commit 与所有源文件 SHA，按 `files` 逐个复制 overlay。默认只允许新增文件；`replaced_base_paths` 明确登记允许替换的基线文件与原 hash。没有文件写入共享源码、其他题目、权威 harness、runtime 或 GitHub。

| Seed | 候选需要处理的初始能力 | 生产代码位置 |
| --- | --- | --- |
| R05_diagnostic_export | 可选完整 SAE codes 诊断导出 | `tasks/capture/wp21_full_codes.py` |
| R07_retired_dataset | `WP21RetiredPrompts` 数据集与 `wp21_retired_prompts` alias | `datasets/wp21_retired_prompts.py` |
| R09_retired_model | 模型扩展 `ext.wp21.retired_gemma`，加载同一真实 Gemma backbone | `models/wp21_retired_gemma.py` |
| R11_retired_task | 通过 Registry 的 `wp21_capture_legacy`，实际委托 CaptureTask | `tasks/capture/wp21_capture_legacy.py` |
| R13_debug_subtask | benchmark 的 debug_norm 与 reconstruction 两个真实计算分支 | `tasks/benchmark/wp21_subtasks.py` 和 `tasks/benchmark/operation.py` |

所有 seed 提供 `tests/wp21/Rxx/before.py`。在该题已安装且外置依赖环境已在分派前准备好后，从本题源码根运行。before.py 将当前源码根放在 Python 搜索路径第一位，不需要逐题 editable install；现有 source-tree alias 负责 saebase.datasets 的导入：

```bash
python tests/wp21/R07/before.py --check-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tests/wp21/R07/before.py --output /absolute/candidate_workspace --device cuda
```

第一条只核对注册、配置和部分读取/调度，输出 `status=registration_only, measured=false`。第二条真实加载 Gemma 并执行；权重、依赖或设备不足应保留错误并记 ENV_BLOCKED。模型扩展 R09 是 model-only，只调用其真实 factory、capture 和 logits；其 provider revision 是源码 SHA，实际 checkpoint 身份在加载后另以 state SHA 记录。

`--output` 必须等于该题 candidate workspace 根目录，脚本在执行前拒绝其它目标；实验存放在其 `experiment/<dataset>/<model>/<task.path>/<sub_name>_<seed>/`。存在旧叶时拒绝覆盖。R13 用 `--subtask debug_norm` 和 `--subtask reconstruction` 分别生成前置证据，task.subtask 真正选择计算，task.path 仅组织目录。

依赖虚拟环境置于工作树之外并在调度前准备，候选源码中不生成 build、egg-info 或虚拟环境目录。每题将真实 before 运行的原始输出、配置及 hash 保存在独立题目工作区，删除操作只作用于安装后的候选源副本。运行后完整检查既有实验字节保持不变。Fresh-process 验证用于排除 Python import cache 与残余 registry 状态；执行者无需也不得通过写私有 registry 字典伪造删除。

## 边界

- R05 是独立 opt-in 导出 wrapper；标准 capture 源码保持基线行为，未预先修复 top_k 或新增批量 capture。
- R07 通过公开 DatasetPlugin/alias 注册与其专属 reader，保持原 dataset identity。现有 built-in prompt reader 的固定清单保持原状；此 seed 没有冒名改成 IOI 来掩盖数据来源。
- R09 只注册一个 ExternalModel；不创建新的模型权重，不删除已有 checkpoint，不更改 built-in Gemma 插件。
- R11 先保留原配置，转成 capture 配置副本后调用真实 CaptureTask；注册键与被委托计算都在结果中记录。
- R13 唯一替换基线文件是 `tasks/benchmark/operation.py` 的入口分支，未选择 subtask 时继续走原 benchmark。删除 debug_norm 后必须保留 reconstruction，不能把整个 dispatcher 都删掉充当完成。

这里的辅助脚本不是独立评分器。正式评分继续读取冻结题目合同，并由可信评审重新运行候选与原始数值核对。
