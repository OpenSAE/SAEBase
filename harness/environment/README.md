# Environment Tree

用观测值而不是猜测，描述 Agent 当前所在物理机器的 CPU、GPU、内存、存储、Python、终端工具和环境限制。

```bash
python harness/environment/probe.py
python harness/environment/probe.py --output harness/environment/out/machine.json
python harness/environment/compare.py harness/environment/out/old.json harness/environment/out/new.json
```

`harness/environment/out/` 中的机器快照只保留在本机，不提交到 Git。查看 `harness/environment/example.json` 的**虚构样例**时必须醒目标注 `synthetic: true`。没有安装 `nvidia-smi` 时 GPU 状态为 unknown，不当作没有 GPU。

执行建议：普通训练默认前台；用户明确要求后台且机器支持时使用 tmux；集群可以有 sbatch，但切换调度方式需用户指定并检查实际调度器。Shell、沙箱和联网能力必须以当前环境探测与实际工具反馈为准。

## 与其他两树的联系

知识节点可以指定 `environment_refs` 以指出依赖 CPU/GPU 等执行条件；环境值是机器条件，不是科学问题的真理。代码树不应推测运行设备，只有实际执行/测量后才能建立关系。

## 环境节点的自然语言表示

公开页面按操作系统、处理器、内存、图形设备、磁盘与运行工具分类；卡片使用“后台终端会话”“Python 解释器”等可阅读标题，不使用命令名或 JSON 片段作为主体。`example.json` 是虚构配置，不是用户机器的实测；真实环境需要在本机执行 `probe.py`。
