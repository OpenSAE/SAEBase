# WP21 综合测试前规范审计

审计基线为 `OpenSAE/SAEBase` 的 commit `41a54cbfdee5cce24d3d8f7517aadef0c752c5ac`，日期 `2026-10-30`。下表行号均指向该原始基线，不随本地修改移动。此次工作是规范、工具和题目的本地准备；本文没有把二十道题的未执行状态计为通过。

## 1. 已有规范的结论

现有 `catalog.json` 有 22 个通用 Contract，版本 `1.1.0`。Always-on 的五项继续生效，数据、配置、资源、恢复、接口、命名空间和来源约束已有定义。WP21 复用它们，通过 [真实工作流协议](../../contracts/REAL_WORKFLOW_PROTOCOL.md) 补充可操作要求，保持历史目录版本与 WP18–WP20 试卷不变。

WP20 的正式结项已经说明其主要验证对象是 Harness 的合成环境，参见原始 [结项记录第 8–10、28–46 行](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/harness/github_work_plan/20_exam_phase_closeout_20261029.md#L8-L46)。本轮因此采用真实 SAEBase 工作树、生产入口和实际产物。

## 2. 发现、分类及本地处理

分类中的“已有规则澄清”指既有 Contract 已有职责，但执行入口或解释存在歧义；“新增操作细则”指本轮预先补齐可执行流程；“实现/验证缺口”指文档无法替代的软件行为或保护措施。

| 发现 | 原始源码依据 | 分类 | 本地处理与剩余边界 |
| --- | --- | --- | --- |
| 旧考试只允许 Harness 沙箱，真实模型和生产修改超出其范围 | [harness/tests/README.md:3–13](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/harness/tests/README.md#L3-L13) | 新增执行细则 | 单列 WP21 独立工作树范围，真实代码改所属生产模块；旧考试授权与历史成绩保持原样 |
| `experiments/` 与 Runtime `experiment/` 表述冲突 | [tests README:32–34](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/harness/tests/README.md#L32-L34)，[experiment_paths.py:50–63](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/saebase/runtime/experiment_paths.py#L50-L63) | 已有规则澄清 | 明确单数为当前正式路径，治理准备过程预建输出根，使用现有 resolver；Runtime 路径实现不变 |
| Dataset/Model skill 指向不存在的 `saebase/runtime/registry.py` | [add-dataset:8](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/.cursor/skills/components/add-dataset/SKILL.md#L8)，[add-model:8](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/.cursor/skills/components/add-model/SKILL.md#L8) | 已有文档修正 | 仅将字面入口修正为实际 `saebase/registry.py` |
| Task skill 指向不存在的 `tasks/runner.py` | [add-task:8](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/.cursor/skills/components/add-task/SKILL.md#L8) | 已有文档修正 | 仅将字面入口修正为实际 `saebase/runner.py` |
| 七类操作、实验 Task ID 与 Subtask 行为未明确区分 | [22-tasks.mdc:9–14](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/.cursor/rules/components/22-tasks.mdc#L9-L14)，[task_components:37–47](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/saebase/runtime/experiment_paths.py#L37-L47) | 新增操作细则 | Task ID 可由既有类别内模块实现；Subtask 必须有 selector、真实分派与错误处理，`task.path` 继续只负责产物层级 |
| 新增技能没有完整删除流程，Registry 没有通用 unregister API | [registry.py:82–137、194–220](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/saebase/registry.py#L82-L220) | 新增操作细则 | 删除题从固定已安装材料独立开始，覆盖实现、注册、别名、配置、调用者和新进程检查；历史数据保留；没有声称新增 unregister API |
| 参数存在于 YAML 不证明参与实际计算 | [CT-CONFIG:323–337](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/harness/contracts/catalog.json#L323-L337)，[Task 边界:13–14](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/.cursor/rules/components/22-tasks.mdc#L13-L14) | 已有合同细化 | 声明类型、默认、合法域、优先级及兼容；用不同合法值和相关反例证明实际算法行为 |
| Hydra multirun 已有，搜索选优和失败 trial 协议不足 | [check_run_location:66–88](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/saebase/runtime/experiment_paths.py#L66-L88)，[CT-RESOURCE:340–354](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/harness/contracts/catalog.json#L340-L354) | 新增操作细则 | 预先冻结网格、seed、预算、指标、方向和 tie-break；保留全部 trial 状态；失败指标不填零，确认评估在选定后运行 |
| Runtime 配置 hash 是 YAML 文本哈希，不能冒充 canonical 内容身份 | [experiment_paths.py:139–154](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/saebase/runtime/experiment_paths.py#L139-L154) | 实现语义澄清 | 保留既有 Runtime 字段，canonical resolved-config hash 另存；源码、配置、数据、checkpoint 身份分开 |
| 实验文档声称记录的信息多于 `run.json` 实际字段，例子还缺必需前缀 | [experiments README:22–32](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/harness/experiments/README.md#L22-L32)，[metadata:144–159](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/saebase/runtime/experiment_paths.py#L144-L159)，[description check:211–218](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/saebase/runtime/experiment_paths.py#L211-L218) | 文档修正与证据补充 | 列清已有字段，真实来源信息由 evidence manifest 补充；示例加 `目标：` 前缀；没有声称原 Runtime 已自动写齐这些字段 |
| 实验 validator 只覆盖 analysis group 结构 | [validate.py:14–46](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/harness/experiments/validate.py#L14-L46) | 验证范围澄清 | 文档标清通过的含义，实际产物、权重、数值和文件完整性由独立真实工作流检查覆盖 |
| 新 WP 编号的 reviewer 目录未自动得到旧 Hook 的全部保护 | [guard_paths.py:80–87](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/.cursor/hooks/guard_paths.py#L80-L87)，[guard_shell.py:61–64](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/.cursor/hooks/guard_shell.py#L61-L64) | 实现/保护缺口 | reviewer 文件放在考生工作树之外；本地 Hook 增加 wp21_scope profile 并泛化 acceptance 保护，依据真实已存在的用户指派激活；环境隔离仍独立落实 |
| Work Order ready 和命令授权不是同一机制 | [Work Order validator:73–101](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/.cursor/skills/workflow/run-work-order/scripts/validate_work_order.py#L73-L101)，[guard_shell.py:242–258](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/.cursor/hooks/guard_shell.py#L242-L258) | 已有权限规则澄清 | 初始状态 `prepared_not_assigned`；直接用户题目分派和冻结 scope 提供执行依据，不伪造人类批准。Hook 兼容问题交治理方处理 |
| 总览中 WP19 状态停留在早期开放范围 | [tests README:44–46](https://github.com/OpenSAE/SAEBase/blob/41a54cbfdee5cce24d3d8f7517aadef0c752c5ac/harness/tests/README.md#L44-L46) | 文档漂移修正 | 改为引用 WP19 专用规范与 Work Plan，避免在总览重复维护过期 readiness 数字 |

## 3. 原有合同如何覆盖本轮

| 本轮行为 | 保留的通用 Contract |
| --- | --- |
| 所有题目 | CT-BASE、CT-SCOPE、CT-TRUTH、CT-CHECK、CT-TRACE |
| 真实执行、来源与设备 | CT-ENV、CT-EVIDENCE、CT-RESOURCE、CT-INTERFACE |
| Dataset/Model/Task 生命周期 | CT-NAMESPACE、CT-INTERFACE、CT-DATA、CT-CONFIG、CT-ERROR |
| 参数及搜索 | CT-CONFIG、CT-DETERMINISM、CT-DATA、CT-RESOURCE、CT-RECOVERY |
| 实验文件、恢复和归档 | CT-EVIDENCE、CT-RECOVERY、CT-ERROR、CT-PRIVACY |
| 源码候选、双语 Context 与审核 | CT-AST、CT-STRUCT、CT-LOCALE、CT-CONTEXT、CT-REVIEW，按题实际输出选择 |

这些是复用关系；逐题的机器可读绑定由 Work Plan 声明并由原 Registry 解析，不以此表代替绑定。

## 4. 本轮规范补充的完成边界

本地更新了真实工作流协议、相关总览和实验文档、三个组件 skill 的失效入口，以及通用规则的最小引用。本地还增加 `.cursor/hooks/wp21_scope.py`，并修改路径/命令 Hook 以识别可信 dispatcher 登记的当前 WP21 指派及泛化 acceptance 保护。它不会因解包或准备而自动授权，也不声称提供 OS 沙箱。原 `catalog.json`、历史评分器、policy、Agent Flow 实现和 Runtime 保持原样。

WP21 使用独立 `saebase.wp21.contract_receipt.v1` schema。旧 trace/acceptance 对 edited_paths 的 Harness 沙箱约束继续服务历史考试；新真实生产路径通过 WP21 scope/evidence 工具核验，并由 reviewer 独立重放。

题目执行仍可能暴露产品实现、环境或规范的新问题。保留现有六类失败归因，另记责任来源和发生阶段；通过基线复现区分 baseline_product 与 candidate，避免把既有软件缺陷自动归为考生错误。规则改进由治理维护者负责，不要求考生改标准使结果变绿。

未纳入本轮修改的非阻塞文档残留包括旧 Core README 的 OpenResearch 网址和历史 Work Order 示例。它们已被识别，真实工作流题目使用本包明确提供的 SAEBase 入口和上下文，不因此扩大为 Atlas 或项目重命名任务。

## 5. 真实源码中保留为考题的行为

以下实现行为已经按原始源码定位；本包没有提前替考生完成这些产品修改。

| 行为 | 原始源码 | 题目处理 |
| --- | --- | --- |
| capture配置top_k但计算固定取10 | conf/task/capture.yaml 与 tasks/capture/operation.py | R02要求实际参数贯通 |
| reconstruction_l2是重建向量范数 | tasks/capture/operation.py | R04另增真实重建误差，保留原字段语义 |
| SAELens训练seed固定0 | tasks/train/backend/training.py | R17新增seed参数并跑1024-token短程训练 |
| artifact_path未检查category符号链接逃逸 | saebase/runtime/experiment_paths.py | R19在授权的工具层实现完整归档/清理边界，Runtime保持受保护；原缺陷未被文档修正冒充修复 |
| benchmark默认loader未转交task.device | tasks/benchmark/operation.py 与 tasks/benchmark/backend/runner.py | R12/R13明确记录实际CPU运行，不能仅凭配置声称GPU通过 |
| 原receipt验证器只接受Harness沙箱编辑 | harness/contracts/trace.py | WP21使用独立收据schema与明确产品scope，原历史验收规则保持 |

最终题包还统一了seed的输出根、外部依赖环境与源码导入方式、固定缓存与实际loader的关系，并保留BLOCKED和ENV_BLOCKED的区别。R15带固定工程目标及独立确认；R16只检验固定网格的中断恢复。
