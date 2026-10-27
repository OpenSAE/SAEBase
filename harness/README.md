# SAEBase Harness

这一目录包含从 OpenResearch 集成的可审计开发框架。SAEBase 原有 `saebase/`、`saebase/data/`、`models/`、`tasks/sae/` 与发布接口继续独立存在。

| 模块 | 作用 |
| --- | --- |
| `core/` | 项目架构、Agent Flow、权限与审批规则 |
| `knowledge/` | 科研知识、状态、来源与 Context 关系 |
| `code/` | Python AST、接口说明及结构 Diff |
| `experiments/` | 实验元数据、目录与可追溯性 |
| `environment/` | 当前硬件环境采集与比较 |
| `transfer/` | 经授权的 B2 / rclone 校验传输 |
| `visualization/` | Core、Knowledge、Code、Environment、Context 五树 Atlas |
| `github_work_plan/` | 明确列出的待审或暂缓工作；不等于执行授权 |

## 合并后的数据与执行边界

- SAEBase 原有 Hugging Face 数据源和 `saebase/data/` 继续保留；来自 OpenResearch 的新增数据集位于 `research_datasets/`，确保 Python 官方 `datasets` 第三方包仍可正确导入。
- 原始 `models/plugins.py` 与 `tasks/plugins.py` 继续提供 SAEBase v0.5 兼容入口。新增 DLBase 注册接口分别为 `models/dlbase_plugins.py` 和 `tasks/dlbase_plugins.py`。
- 原始 `conf/config.yaml` 不变；新 `dlbase.train` 使用 `conf/openresearch.yaml`。
- 本目录引入的 OpenResearch Core/Knowledge 数据是迁移参考，尚未按 SAEBase 具体实现与科研证据逐项重审。生成图不代表 GPU 或实验结论已获验证。

## 可以立即执行的检查

```bash
python .cursor/hooks/validate_repo.py --full
python -m pytest -q tests/test_openresearch_integration.py
python -m dlbase.train --cfg job
python harness/code/build.py
python harness/visualization/export.py
python harness/visualization/build_pages.py
python harness/visualization/build_pages.py --check
```

这些命令是目标验收命令。是否通过应以 [PR #60](https://github.com/OpenSAE/SAEBase/pull/60) 的实际 CI 和日志为准，不能仅凭本 README 认定成功。

## 需要外部条件的验收

[SAEBase × OpenResearch 外部实测计划](github_work_plan/SAEBASE-OPENRESEARCH-EXTERNAL-ACCEPTANCE-DEFERRED.md) 记录需要真实 GPU 权重、外部数据授权、目标机器或发布权限的检查。文件状态为 **DEFERRED**，不是已激活的工作命令，也不授权自动下载权重、执行长训练或生产部署。

## Agent 权限

遵循 `.cursor/hooks.json`、`.cursor/policy.json` 与 `core/AGENT_FLOW.md`。未获得人类明确批准的提案不应直接修改 Core、规则、代码或科研状态。Hooks 是开发工具级限制，不等同于操作系统隔离。
