# Backblaze B2 / rclone 可审计传输

本模块已存在，默认 **dry-run**，只有 `--execute` 才会连接 B2。用户在服务器上负责预配置 HF、GitHub 和 B2，程序只通过 rclone 的私有配置读取凭据，不记录密钥。不执行 `sync`、`purge`、`delete`、系统目录同步，也不覆盖身份不一致的已有文件。

## 推荐起始配置

B2 Native API `rclone` remote。混合科研数据推荐 `transfers=16`、`checkers=16`、`multi-thread-streams=4`、`multi-thread-cutoff=256M`、`chunk-size=64M`、`fast-list`、`retries=10`、`low-level-retries=20`。上传大文件另行使用 B2 原生 `b2-upload-concurrency=4`。这些是起始值，实际速度受带宽、文件分布、磁盘和 B2 限流影响；可以在既有配置范围内调节。16×4×64 MiB 理论上可带来数 GiB 上传缓冲压力，不能假定所有主机足够；低 RAM 机器减少 transfers / concurrency / chunk size。

## 上传、校验、恢复

```bash
# 默认预览，不访问远程
python harness/transfer/b2.py push --local experiment/_studies/example \
  --remote b2:bucket/saebase/WP23/run_001
# 在 Work Plan 已授权该唯一前缀后，允许传输并留存不可变收据
python harness/transfer/b2.py push --local experiment/_studies/example \
  --remote b2:bucket/saebase/WP23/run_001 \
  --execute --receipt experiment/_studies/example/reports/run_001/b2-copy.json
# 只读检查。rclone check 可用的 B2 SHA1/hash 或文件属性不能代替独立 SHA256。
python harness/transfer/b2.py verify --local experiment/_studies/example \
  --remote b2:bucket/saebase/WP23/run_001 \
  --execute --receipt experiment/_studies/example/reports/run_001/b2-check.json
# 迁移到新服务器时只复制至目标工作空间（copy --immutable）。
python harness/transfer/b2.py pull --local /workspace/restore/WP23/run_001 \
  --remote b2:bucket/saebase/WP23/run_001 --execute
```

所有正式 WP 交付应有 `artifact_manifest.json`、本地 SHA256、实际 B2 prefix、复制与校验返回码、耗时及缺失清单。目录范围由 WP 授权，发布受限原始题干、模型权重或私有数据不因配置好 B2 而自动允许。传输结束仍需通用 Audit、独立科研验收；未执行的远程任务保持 `NOT_RUN`。

参考 <https://rclone.org/b2/>、<https://rclone.org/docs/>、<https://rclone.org/commands/rclone_copy/>、<https://www.backblaze.com/docs/cloud-storage-integrate-rclone-with-backblaze-b2>。
