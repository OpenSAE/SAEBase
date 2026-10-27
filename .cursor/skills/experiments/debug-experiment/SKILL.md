---
name: debug-experiment
description: 当实验无法启动、训练报错、数据不匹配、指标异常、需要 smoke 或定位性能问题时使用。
---

# 实验调试与最小化复现

1. 先读取实际错误、运行命令、Hydra 组合与 Python 解释器；不得仅凭 README 猜测实现。
2. 根据问题归属定位 dataset policy、model compatibility、task pipeline checks 或 dlbase 通用入口。先修局部模块。
3. 优先使用合成数据和极短 smoke，检查 shape、label、dtype、loss/metric 语义及 batch 边界，区分配置失败、准入失败与 GPU 运行失败。
4. 新增可执行调试工具放 scripts/，自动测试放 tests/；不得写入根目录或将测试脚本变成生产依赖。
5. 使用临时输出目录，避免覆盖既有实验 checkpoint；必要时设置 DLBASE_SMOKE 防止自动拉起 TensorBoard。
6. 修改后运行 python .cursor/hooks/validate_repo.py --full，加上与本次 bug 对应的回归验证。
7. 汇报复现步骤、根因、修复内容、回归结果以及未消除的风险。
