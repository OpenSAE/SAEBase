# Core · 实现边界

## 分工

1. **Core** 只管理全项目架构、开发目标、模块间标准与已有实现的状态。其图 `project_graph.json` 不能作为数据集/模型的科学理论。
2. **Knowledge** 维护 SAE 特征捕获、干预、解释、评测与训练问题，权威信息在 `doc/knowledge/`，研究结果依据真实实验单独验证。
3. **Code** 生成 AST 类/函数图，派生数据放 `doc/code/`；`core_ids` 和 `context_ids` 分别引用两棵权威图。不能让人工图直接覆盖真实源码。
4. **Environment** 由设备现场探测；真实值与虚构环境示例严格分离。
5. **Experiments** 的生产协议写在 `saebase/runtime/experiment_paths.py` 和 `conf/`，Harness 只验证/解释这些产物，不能成为训练主循环依赖。
6. **Transfer** 必须明确指定来源、目标，默认不传输，真实 B2 同步需要连接用户账户；不要把凭据提交 Git。
7. **Visualization** 是四棵树+术语字典的公共只读前端。生成静态数据后再发布公开镜像，不自动暴露私有机器信息。

## 工作过程

- GitHub 端提出新科研问题或架构任务，优先查 Core 与 Knowledge 中已存在的概念和术语。
- 用当前 Work Order 限定本地 Agent 实际可以加载的 node IDs 和模块。
- 需要改代码，先登记预期 Code Tree 变化；执行后重新跑 AST Diff。
- 训练后根据配置、checkpoint、独立测试和环境真实记录判断知识目标状态；代码存在不能证明结果达标。
- 修改 Core / Knowledge / Glossary 状态时提交证据与说明，未复审不直接写 verified。

## 风险和局限

静态 Atlas 不代表真实训练通过；验证集与测试集若复用相同样本，则不能当作独立测试证据；HTML 公网镜像不是实时读私有仓库；`.cursorignore` 不是 OS 沙箱。与 B2、多 GPU 和真实训练相关的端到端验收仍需对应环境实测。

## 仓库文件架构纲领

在修改之前先确定文件所有权。改动权限与架构合规是两个独立条件。新增实体必须说明为什么现有模块不足以满足需求。宏观框架与微观实验分离。

| 所有者 | 合法职责 |
| --- | --- |
| `saebase/` | 通用 API、Registry、Runtime 与跨任务机制 |
| `datasets/<Dataset>/` | 数据来源、适配器；对应配置在 `conf/dataset/` |
| `models/<Model>/` | 模型格式、加载与适配；配置在 `conf/model/` |
| `tasks/<Task>/` | 可跨数据集、模型、实验重用的任务；配置在 `conf/task/` |
| `scripts/experiments/<Study>/` | 具体研究执行与组合；配置在 `conf/experiment/<Study>/` |
| `scripts/` | 可执行脚本与启动入口 |
| `tests/` | 产品功能及集成验证 |
| `experiment/` | Runtime 管理的运行产物 |
| `harness/` | 治理、审阅、知识和独立验收 |
| `.github/` | GitHub 原生配置，与 `harness/github/` 独立 |

**依赖方向**：具体实验可以组合 Dataset、Model、Task，但通用组件不得反向依赖某个具体 Experiment。Dataset 专属解析属于对应 Dataset，不能把单一研究提升为 Dataset 总纲；启动脚本不能散落在组件目录中。配置、可执行入口、实现和结果必须分别归其所有者。

**Scripts / Tests 开放组织原则**：优先按 Work Plan，例如 `scripts/workplan/wp22.sh` 与 `tests/workplan/wp22/`；或者参照 Runtime 实验路径，在一级目录内部按 `<dataset>/<model>/<task>/<sub_name>/` 分类。研究级布局 `scripts/experiments/<study>/`、`tests/experiments/<study>/` 也可接受。它们是优先约定，不是唯一合法层级；其他方案须交代归属、复用依据和为何现有布局不足。不要将临时脚本或测试直接散落在仓库根目录。

**新增结构审批**：现有模块内新增受控实例可按照已授权 Work Plan 进行；新增仓库一级目录、通用 Registry、独立配置体系、跨模块框架或改变依赖方向必须单独提出架构变更并通过独立审阅。每次 PR 列出新增/移动文件的所有权、依赖、配置和注册影响、测试及证据。Path scope 授权不代表架构许可。机器检查处理结构性错误，Reviewer 审核语义正确性。历史冻结验收和科研证据不因迁移重写。


**一级目录扩展规则**：不再接受新建顶层 `experiments/` 科研源码包。具体实验由既有 `conf/experiment/<study>/`、`scripts/experiments/<study>/` 和受控产品测试组织；运行记录归单数 `experiment/`。新增其他根目录必须经过独立架构审批。
