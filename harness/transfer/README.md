# Backblaze B2 / rclone 传输模块

大文件、checkpoint 和实验产物的备份属于可选基础设施。**默认只输出 dry-run 计划**，没有用户明确操作时不调用远程服务、不读取密钥、不上传/下载。

```bash
# 先本机安装 rclone 并在用户配置区定义 b2remote。不要将密钥写入项目或 Work Order。
python harness/transfer/b2.py push --local experiment/CUB_200 --remote b2remote:my-bucket/openresearch/CUB_200
# 用户确认后才真正传输：
python harness/transfer/b2.py push --local experiment/CUB_200 --remote b2remote:my-bucket/openresearch/CUB_200 --execute
# 传输后只读校验：
python harness/transfer/b2.py verify --local experiment/CUB_200 --remote b2remote:my-bucket/openresearch/CUB_200 --execute
# 需要从 B2 恢复时：
python harness/transfer/b2.py pull --local ./restore/CUB_200 --remote b2remote:my-bucket/openresearch/CUB_200 --execute
```

使用 `rclone copy` 而非 `sync`，**不会主动删除目标端的额外文件**；也不使用 `purge`。可通过重复同一个 `copy` 任务实现增量传输/失败重试。完整性校验使用 `rclone check --one-way`，依据后端可用的校验和；大文件的哈希支持受后端能力约束。命令没有 `--execute` 时只显示计划，避免误触发传输。

性能：B2 大文件通常采用 multipart；`--transfers` 与分块大小共同影响 RAM 占用，请以 `harness/environment/probe.py` 观测到的内存/GPU 负载调节并行度。不要盲目把传输并发调到极大。

参考：<https://rclone.org/b2/>、<https://rclone.org/commands/rclone_check/>。
