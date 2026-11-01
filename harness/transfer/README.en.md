# Backblaze B2 and rclone transfers

Use the existing **B2 Native API** rclone remote already configured on each user server. The CLI defaults to preview and requires explicit `--execute` for a real transfer. It never reads, prints or writes keys to the repository. The tool always uses `copy --immutable`, never `sync` or `purge`; a changed existing destination fails rather than being overwritten.

Initial mixed-research-data settings are 16 transfers, 16 checkers, four multithread streams, 256M cutoff, 64M chunk, fast-list, ten retries and twenty low-level retries. Multipart B2 upload concurrency is separate, initially four chunks per file. Tune for RAM, object sizes, throughput and rate limiting; 16 × 4 × 64 MiB can consume several GiB of buffer memory.

```bash
python harness/transfer/b2.py push --local experiment/_studies/example --remote b2:bucket/saebase/WP23/run_001
python harness/transfer/b2.py push --local experiment/_studies/example --remote b2:bucket/saebase/WP23/run_001 --execute --receipt experiment/_studies/example/reports/run_001/b2-copy.json
python harness/transfer/b2.py verify --local experiment/_studies/example --remote b2:bucket/saebase/WP23/run_001 --execute --receipt experiment/_studies/example/reports/run_001/b2-check.json
python harness/transfer/b2.py pull --local /workspace/restore/WP23/run_001 --remote b2:bucket/saebase/WP23/run_001 --execute
```

The Work Plan must authorize the exact source, destination prefix and content types, and its primary filled report must link transfer receipts. The remote rclone check is not a substitute for a locally computed SHA256 manifest or scientific acceptance. Missing credentials or access are BLOCKED, not fake success.

Documentation: <https://rclone.org/b2/>, <https://rclone.org/docs/>, and <https://www.backblaze.com/docs/cloud-storage-integrate-rclone-with-backblaze-b2>.
