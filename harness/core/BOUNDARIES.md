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

静态 Atlas 不代表真实训练通过；默认 CUB val=test 的历史评测协议不能当作独立测试证据；HTML 公网镜像不是实时读私有仓库；`.cursorignore` 不是 OS 沙箱。与 B2、多 GPU 和真实训练相关的端到端验收仍需对应环境实测。
