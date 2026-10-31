# SAEBase 任务规划与本地 Agent 交付协议

## 原则与所有权

本协议连接现有 GitHub Work Plan、Work Order v3、Universal Task Contracts、
Agent Flow、本地 Cursor Agent、GitHub PR 和独立验收。它**不**改变
`harness/core/BOUNDARIES.md` 的文件归属、七类 Task、Registry、Runtime
或科研证据标准。GitHub 侧负责设计、代码准备及独立验收，
本地 Agent 在用户指派后执行环境检查、代码改动和获批的真实实验。

入口为 `.github/skills/saebase-delivery/SKILL.md`，
本地入口为 `.cursor/skills/workflow/saebase-delivery/SKILL.md`。
原 `run-work-order`、Hooks、Agent Flow、Contracts 和冻结试卷仍是权威。

## 计划与任务拆分

每个任务先声明结果和可验证终点，再定位已有实现。
新增/移动文件逐项说明 owner、可复用模块、配置、依赖方向及兼容影响。
按依赖、授权和证据边界划分任务；小改动优先一个 PR，
多个 Agent 必须有不重叠的工作树、明确 base SHA 和单一集成人。
不要仅为每个阶段拆 PR，也不允许研究实例升格为新框架。

计划至少具有以下信息，并继续使用现有 Work Plan 的章节标准。

| 字段 | 用途 |
| --- | --- |
| Goal / Result | 可观察的目标和排除项 |
| Branch / Base SHA | 精确源码身份、单分支 origin fetch 说明 |
| Ownership / Scope | 复用、允许更改的路径及保护范围 |
| DAG / Agent | 先后依赖、并行条件、集成人 |
| Contracts / Authority | 既有 Contract ID 与真实审批主体 |
| Precheck | 每个阶段需要的安全检查、成功和停止条件 |
| Environment / Pilot | Python、依赖、资源、最小真实操作及预算 |
| Evidence / Handoff | 命令、退出码、输入/模型/代码哈希、未运行事项 |
| Merge / Release | 独立重放、精确 SHA 的 CI 和发布条件 |

## Precheck 选择与执行

首先使用不写文件的分类器：

```bash
python scripts/workplan/delivery_precheck.py --mode plan --profile experiment
python scripts/workplan/delivery_precheck.py --mode plan --changed tasks/benchmark/example.py
```

上述命令只能提出检查建议，不声称运行了依赖、Pilot 或 GPU。
本地用户已指派的生产任务采用：

```bash
python scripts/workplan/delivery_precheck.py --mode local \
  --expected-branch YOUR_BRANCH --expected-sha YOUR_SHA \
  --work-order YOUR_ORDER_ID --run-safe
```

`--run-safe` 仅执行固定的只读命令，不自动 pip install、git fetch、
checkout、清理、下载或启动 GPU；需要额外计算和远程动作时先核验权限。
局部考试沿用 `harness/tests/` 的现有例外和独立 reviewer，不因为
此脚本的存在赋予生产修改权限。

选择检查时遵循最小原则：
文档看引用；代码看语法、依赖及相关测试；Dataset/Model/Task
看 Registry 与配置；Experiment 在真实环境中检查依赖、资产、预算、
最小 CPU 和获批的真实 Pilot；治理及 release 需要独立审批。
这里的分类不是替代已有 Validator 的新验收器。
实际测试通过与源代码可编译、配置可合成是不同证据级别。

## 状态与交付

- `PASS` 只代表实际执行的检查成功。
- `FAIL` 表示实际命令非零退出或身份检查失败。
- `BLOCKED` 表示缺少授权、Work Order、环境、资源或固定身份。
- `NOT_RUN` 表示建议执行但当前阶段尚未执行；不得虚构结果。

本地 Agent 原样保留命令、退出码、设备与实际输出，报告源码
base/head SHA、修改文件、测试结果、剩余失败和权限需求。实验数据及
真实权重由原 Runtime 的 `experiment/` 和既定证据协议管理。
GitHub 审阅方按同一源码身份独立重放测试，报告来源、审批和
科研有效性，避免把 Agent 自评写成最终验收。

## GitHub 交付优化

相关文件以一个或少量完整 Git Commit 批量推送，不逐文件建立 PR；
Agent 分支依赖必须显式记录，未经授权不得向 `main` 合并。
必要的 CI Job 名称、真实 GPU 验收与发布的精确 main SHA 保持不变。
在 PR 阶段验证 Atlas 生成产物及网站打包清单，避免 post-merge
Snapshot Bot 更新 main SHA 后使原来的发布 CI 身份失效。
对尚未实现的 Atlas 预合并生成门槛，明确记录为后续工程优化，
不能将此文档当作已完成的自动保证。

## 失败示例

- Python 依赖 resolver 冲突：报告冲突链、Python 版本、错误日志；
  停止 Pilot，交由授权维护者修复。
- Work Order 不存在或不匹配：`BLOCKED`，不能通过复制模板授权自己。
- 本地 GPU 不符合预算：`BLOCKED`，不私自换模型、改变样本或量化。
- GitHub Atlas Bot 推送新 SHA：发布 gate 必须审查**最新** main SHA，
  不复用旧 SHA 的成功 CI，也不绕过发布策略。
