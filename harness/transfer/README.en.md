# Backblaze B2 and rclone Transfers

Backing up large files, checkpoints, and experiment outputs is optional infrastructure. **The default operation is a dry run.** Without an explicit user-approved execution, the tool does not contact the remote service, read credentials, upload, or download.

```bash
# Install rclone locally and configure b2remote in the user's private configuration.
# Never put credentials in the repository or a Work Order.
python harness/transfer/b2.py push --local experiment/IOI --remote b2remote:my-bucket/openresearch/IOI
# Transfer only after explicit authorization:
python harness/transfer/b2.py push --local experiment/IOI --remote b2remote:my-bucket/openresearch/IOI --execute
# Read-only integrity verification:
python harness/transfer/b2.py verify --local experiment/IOI --remote b2remote:my-bucket/openresearch/IOI --execute
# Recover previously stored artifacts:
python harness/transfer/b2.py pull --local ./restore/IOI --remote b2remote:my-bucket/openresearch/IOI --execute
```

The implementation uses `rclone copy`, rather than `sync`, so extra objects at the destination are not deliberately deleted. It does not call `purge`. Repeating an authorized copy can resume or incrementally update a transfer.

Integrity checks use `rclone check --one-way` with checksums supported by the chosen backend. Large-file hash verification depends on backend capabilities. Without `--execute`, commands display a plan instead of transferring.

Backblaze B2 commonly uses multipart uploads for large files. Transfer concurrency and chunk size jointly affect RAM use. Choose values based on observed local resources from `harness/environment/probe.py`, rather than assuming maximum parallelism is safe.

References: <https://rclone.org/b2/> and <https://rclone.org/commands/rclone_check/>.
