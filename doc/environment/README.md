# Environment — 本机环境来源

Environment 描述**当前执行设备**的运行条件，而非仓库中推断的事实。

权威入口：[`harness/environment/README.md`](../../harness/environment/README.md) 和 [`probe.py`](../../harness/environment/probe.py)。

仅在获得本机探测授权时，运行 `python harness/environment/probe.py`；保存的快照需有 `captured_at_utc` 与 `synthetic` 状态。未读或旧设备的快照不能作为当前环境证据。`harness/environment/example.json` 是合成示例；缺失 `nvidia-smi` 表示 GPU unknown，不表示 absent。

不要公开本机私有快照、凭据或绝对路径。对话读取器仅使用当前任务明确允许的精确快照文件，不自动探测硬件。
