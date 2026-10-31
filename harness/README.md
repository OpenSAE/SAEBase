# SAEBase Research Harness

Harness 为 SAEBase 提供可审计的科研代码导航、知识管理、环境记录和 Agent 审阅流程。
Research Lab、Dataset、Model、Task 及通用 SAE 接口均由 SAEBase 仓库统一维护。

**当前 Harness 基线（WP21.2 已结项）**：[WP21 阶段结项及旧 Harness 兼容验证](github/work_plan/21_harness_phase_closeout_20261030.md)。WP18–WP20 的历史 Reference/Verifier 和 Contract 不变；WP21 真实工作流考试的工具、审计修复、round01/round02 记录已统一归档。入口见 [Harness Tests](tests/README.md)，冻结题目见 [WP21](tests/wp21/README.md)。产品代码与真实 GPU 问题独立处理。

| 模块 | 作用 |
| --- | --- |
| `core/` | 项目结构、Agent Flow 与审批 |
| `contracts/` | 先验通用 Task Contracts、解析器、验证不变量及执行收据 |
| `tests/` | Harness 本地 Agent 测验运行、候选 Tree、独立验收分离 |
| `knowledge/` | 科研知识、来源与 Context 关联 |
| `code/` | 真实 Python AST、接口说明和差异 |
| `experiments/` | 产物路径、配置与来源验证 |
| `environment/` | 本地环境记录与资源比较 |
| `transfer/` | 明确授权后的数据校验传输 |
| `visualization/` | Core、Knowledge、Code、Environment、Context 五棵 Atlas Tree |
| `github/work_plan/` | 历史记录、待审任务与外部验收计划 |
| `github/development_doc/` | SAEBase 开发文档与公开说明 |

## 代码边界

- `datasets/`、`models/` 和 `tasks/` 按具体数据集、模型和七类研究操作直接组织。
- `saebase/registry.py` 与 `saebase/runner.py` 是唯一注册和分派入口。
- `saebase/runtime/` 负责通用实验运行、种子、配置和产物语义。
- `saebase/data/` 保留已发布 API 所需的兼容性导入；Dataset 源码在 `datasets/`。
- `saebase/research_lab/` 提供公开数据浏览及明确受控的发布流程。
- 现有发布标签、实验数据与科研结果不因源码迁移而自动获得新的验收状态。

## 离线校验

```bash
python .cursor/hooks/validate_repo.py --full
python -m pytest -q harness/github/work_test/test_harness_migration.py harness/github/work_test/test_openresearch_integration.py
python -m saebase.runtime.train --cfg job
python harness/core/validate.py
python harness/knowledge/validate.py
python harness/code/build.py
python harness/visualization/export.py
python harness/visualization/build_pages.py --check
```

实际通过情况必须以对应 commit 的 CI 结果为准。需要 GPU、权重、私有数据审核或生产权限的验收见
[外部验收记录](github/work_plan/SAEBASE-OPENRESEARCH-EXTERNAL-ACCEPTANCE-DEFERRED.md)，不能由离线检查替代。

## Agent 权限

遵循 `.cursor/hooks.json`、`.cursor/policy.json` 与 `core/AGENT_FLOW.md`。
未经人类批准不得修改治理层，批准新增节点也不表示科研结果已经验证。

## GitHub Agent 与文件所有权

从 [GitHub 工程入口](github/README.md) 读取 Work Plan、开发文档、独立测试。正式编辑前须遵守 [架构纲领](core/BOUNDARIES.md)，确定文件归属、层次及新增实体的必要性。Scripts 和 Tests 优先按照 Work Plan 或 Dataset/Model/Task/Sub-name 分类；具体规范允许有依据的扩展。
