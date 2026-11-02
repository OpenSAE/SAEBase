# WP23 · Agent Ready 交付审计

**判定：AGENT READY FOR LOCAL VALIDATION（软件交付已就绪，真实运行未验收）。**

## 交付边界

WP23 是独立、通用的 SAEBase Dataset / Model / SAE / Compatibility / Task Precheck。它自己定义检查范围与配置，不从别的 Workplan 继承协议。GitHub 的责任是提供能执行检查的产品代码、实际 CLI 命令及离线软件测试；Local Agent 的责任是实际运行这些检查，检验 Resume、异常、收据、Audit 和真实资源适配。

“Agent Ready”不等于所有 Model、Dataset、SAE、Benchmark、Resume 已 PASS，更不等于科学结论。失效、超时、缺权重或缺许可都应成为如实记录的本地测试结果；不允许提前虚构成功。

## 已实现的交付项

| 条目 | 现状 | GitHub 证据 |
| --- | --- | --- |
| 独立入口 | `bash scripts/workplan/wp023.sh` 有 setup / inspect / inventory / asset_prepare / dataset / model / sae / compatibility / task / audit / handoff / run / resume | Source SHA `37b7b4f96b77b60a908deef4fce59e65ac89e18a` |
| 默认配置 | `conf/experiment/wp23_precheck.yaml` 定义自身目标规则、每 Pair 两条输入、最大 128 tokens / 180 seconds；入口拒绝与已实现常量不一致的配置 | 同源配置校验入口 |
| 用户权限 | 默认无 GPU、无联网或下载、无发布；计算需要显式 `--allow-compute --allow-gpu`，网络/安装/下载各自显式开启 | `wp023.sh` |
| 组件 | 独立 Dataset、Model Forward、SAE 和合法组合检查器已存在，实际权重验证在本地执行 | `tasks/precheck/`、`scripts/workplan/wp023_orchestrator.py` |
| 收据与审计 | 每目标状态、四张 CSV、运行 manifest、Audit 和填写的 Workplan 路径已经实现 | `audit_and_handoff` |
| 质量 | GitHub PR #158 / #164 / #165 的 Python 质量、Product 和 Harness 软件 CI 成功；这些不是 GPU 实测 | <https://github.com/OpenSAE/SAEBase/actions/runs/41415994491> |
| Atlas | 六域复核表明没有当前要写入的长期节点描述/性质/关系修改，所以无需本轮 Node Human Check | `reviews/WP23/node_impact.json` |

## 明确留给 Local Agent 的测试与风险

- **真实 Precheck：**全部当前声明的 Dataset/Model/SAE Site、合法组合及有界真实 Task，逐项输出 PASS/FAIL/BLOCKED/NOT_RUN 和实际资源成本。
- **Benchmark：**实际任务需要真实、有限的测量配置；当前默认可能明确报 `BLOCKED_CONFIGURATION`。这个结果须由 Local Agent 观察、记录并报告，不允许标为已经通过。
- **Resume：**检查未完成中断、已验证结果复用及改变权限/资源后的重新运行。当前脚本对旧 `BLOCKED_AUTH` 采用保守策略：在权限变更时要求新 run ID，而不是错误复用旧阻塞；更细粒度恢复能力在真实测试中评价。
- **Audit：**验证缺失、篡改、重复收据、不同 source/config SHA、部分阻塞的四表覆盖与报告生成。代码与单元测试存在不代表这些场景已经本地验证。
- **权限：**本任务未自动授权真实 GPU、联网下载、受限资产、付费运行、Native 训练、B2 或公开发布。需要这些操作时，Local Agent 应按任务合同申请明确许可，缺少许可则记录 BLOCKED。

## 给 Local Agent 的入口

首先同步确切源码，阅读 `AGENTS.md` 和 WP23 正式 Workplan。推荐先执行：

```bash
bash scripts/workplan/wp023.sh setup
bash scripts/workplan/wp023.sh inspect
bash scripts/workplan/wp023.sh run --run-id wp23_run_01
```

在明确获得 GPU/资源授权及准备实际资产之后，用**新的** run ID 运行：

```bash
bash scripts/workplan/wp023.sh run --run-id wp23_gpu_01 --allow-compute --allow-gpu
```

本次交付的完成定义是：所有入口和默认软件能力已由 GitHub 准备，真实运行、恢复和异常验证作为本地工作正式交接。Local Agent 不需自创脚本；真实执行暴露的缺陷应提交清晰收据，由 GitHub 侧按通用防复发规范继续修复，不得由 Agent 自行改变科学规则。
