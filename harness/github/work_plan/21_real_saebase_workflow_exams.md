# WP21 真实 SAEBase 工作流综合考试

本工作单准备20道可分派给20个agent的独立综合题。所有题目基于固定源码 `41a54cbfdee5cce24d3d8f7517aadef0c752c5ac`，机器题库为 [cases.json](../../tests/wp21/cases.json)，具体运行义务见 [真实工作流协议](../../contracts/REAL_WORKFLOW_PROTOCOL.md)。本地交付状态为 **DESIGN_READY，WP21.1 EXAM_SYSTEM_READY**。尚未执行20位考生的真实考试，也未完成20道题的独立真实checkpoint验收。

WP21.1 已新增冻结完整性校验、单题预算限制、真实 loader 探测、独立工作树 Reviewer 及逐项复跑记录，执行说明见 [EXAM_READY.md](../../tests/wp21/EXAM_READY.md)。此状态只表示考试基础设施可部署，真实 checkpoint 环境仍需逐项准备。

## 1. 本轮目标

考生必须实际利用SAEBase完成真实小实验或功能生命周期操作，交付可复算结果。能力覆盖实验基线、代码功能增删、dataset/model/task/subtask的新增与退役、参数贯通、参数搜索、故障恢复以及experiment文件完整性和归档管理。

沿用现有22个通用Contract。题目选择已有ID，协议细化其在真实运行中的含义；不为每题另外发明一份独立通用规范。当前公开七类研究任务与Gemma-2 2B主线保持。R08新增真实Pythia14m限定在考试external provider作用域；R09只删除临时Gemma provider。所有source修复由考生完成，题包不会预先修复top_k、seed等考试目标。

## 2. 审计完成后采用的规范

1. 每题固定真实入口、当前已存在行为与待实现行为；legacy saebase CLI、runtime.train、public API分别声明，不将configured_only算作执行。
2. 真实模型加载、实际运算、输入与文件身份连成证据链。synthetic软件检查可以辅助，完整正例使用固定pretrained checkpoint。
3. 增删组件验收配置、registry、reader/loader、任务调用、产物与fresh process；删除组件保留历史数据及结果。
4. subtask必须真正dispatch并计算，task.path单独管理目录；新增参数验证类型、范围、默认兼容和数值影响。
5. 搜索先冻结参数全集和输入用途，再逐trial记录实际命令、状态、退出码及原始结果。replacement值与scale系数的含义分别声明。
6. 重试使用新attempt并引用已有不可变结果。仅验证身份相同的完成trial可以复用；失败日志不覆盖。
7. experiment遵守已有路径与typed artifact目录。新canonical配置hash与runtime现有hash并列，不改写旧身份语义。
8. 考生自测、证据结构检查、独立真实复跑、人工正式审批是不同状态。题包工具负责预检/分派/记录/完整性，不把这些工作冒充数值行为grader。

审计原始证据与具体缺口见 [AUDIT.md](../../tests/wp21/AUDIT.md)。本工作单和协议优先界定这一次隔离本地考试的操作范围，不使历史工作单或公开发布状态自动改变。

## 3. 20题分派矩阵

| 题号 | 内容 | profile | 自给定seed | 最大trial |
| --- | --- | --- | --- | ---: |
| [R01](../../tests/wp21/cases/R01.md) | 真实采集、恒等写回与特征消融 | gemma_inference | 无 | 1 |
| [R02](../../tests/wp21/cases/R02.md) | 让 capture 的 top_k 参数真实生效 | gemma_inference | 无 | 6 |
| [R03](../../tests/wp21/cases/R03.md) | 新增真实 token_position 参数 | gemma_inference | 无 | 4 |
| [R04](../../tests/wp21/cases/R04.md) | 增加真实重建误差并保持旧指标含义 | gemma_inference | 无 | 4 |
| [R05](../../tests/wp21/cases/R05.md) | 删除临时完整 code 导出功能 | gemma_inference | R05_diagnostic_export | 2 |
| [R06](../../tests/wp21/cases/R06.md) | 挂载带明确划分的新数据集 | gemma_inference | 无 | 4 |
| [R07](../../tests/wp21/cases/R07.md) | 删除临时数据集并保留历史实验 | gemma_inference | R07_retired_dataset | 2 |
| [R08](../../tests/wp21/cases/R08.md) | 挂载真正的新 pretrained 模型 | pythia14m_inference | 无 | 1 |
| [R09](../../tests/wp21/cases/R09.md) | 删除临时模型 provider | gemma_inference | R09_retired_model | 2 |
| [R10](../../tests/wp21/cases/R10.md) | 新增已注册的 feature summary 任务 | gemma_inference | 无 | 1 |
| [R11](../../tests/wp21/cases/R11.md) | 退役临时已注册 task | gemma_inference | R11_retired_task | 2 |
| [R12](../../tests/wp21/cases/R12.md) | 新增实际执行的 reconstruction subtask | gemma_inference | 无 | 5 |
| [R13](../../tests/wp21/cases/R13.md) | 删除 debug_norm 子任务并保留兄弟分支 | gemma_inference | R13_debug_subtask | 4 |
| [R14](../../tests/wp21/cases/R14.md) | 让 capture 按记录批量消费输入 | gemma_inference | 无 | 3 |
| [R15](../../tests/wp21/cases/R15.md) | 十二组 replacement 搜索与独立确认 | gemma_inference | 无 | 14 |
| [R16](../../tests/wp21/cases/R16.md) | 中断后恢复真实六组搜索 | gemma_inference | 无 | 6 |
| [R17](../../tests/wp21/cases/R17.md) | 训练 seed 贯通与短程 SAE 复现 | gemma_training | 无 | 3 |
| [R18](../../tests/wp21/cases/R18.md) | 真实实验产物完整性与稳定配置身份 | gemma_inference | 无 | 1 |
| [R19](../../tests/wp21/cases/R19.md) | 安全归档、精确清理与恢复一个真实 run | gemma_inference | 无 | 1 |
| [R20](../../tests/wp21/cases/R20.md) | 真实多阶段实验与故障后交接 | gemma_inference | 无 | 3 |

每个agent只领取一题。每题使用同一baseline与自身seed，不要求完成任何前置题。R05/R07/R09/R11/R13的seed提供真实可执行的待删除能力，避免要求考生先凭空构建删除对象。可以先分派4–5题检查环境和证据流程，再独立并行其余题；总题目数保持20。

## 4. 冻结资源与输入

- common输入为 [prompts.jsonl](../../tests/wp21/fixtures/common/prompts.jsonl)，24条、12组，每组2条；train12条属于group-001–group-006，validation6条属于group-007–group-009，test6条属于group-010–group-012。规范字段是sample_id、prompt_group_id、split、prompt。它们是原创工程输入。
- `gemma_inference` 使用冻结真实Gemma-2 2B及canonical layer25 SAE；`gemma_training` 使用同一真实模型及已锁定原生SAELens环境，训练产生的新SAE另记identity。
- `pythia14m_inference` 使用官方revision `d31f659f077f9ea500c4d7faa999893d7f28cbdf`，只验证model-only external provider。R09也可全程model-only，SAE身份明确为未使用。
- 具体本地snapshot、实际设备、文件hash与依赖版本由环境预检锁定。未取得这些资源时保持ENV_BLOCKED，不自动换模型、伪造GPU或把toy升级为最终证据。
- R17固定每次training_tokens=16×64=1024，共3次最多3072。ctx_len=16不再乘到SAELens训练token数中。训练只使用train原始记录，必要的确定性重复有ID映射，validation/test不混入。
- R02固定六个真实配置运行，覆盖错误baseline、修复后3/7/10、省略默认与大于d_sae的边界；非法值在加载前拒绝并保留日志。
- R15先完成12条validation网格，以两条validation的平均logit_delta_l2最小化选择6个feature/value候选之一，平局按feature_id/value升序；冻结best_config后再做两条heldout确认，共14次干预。R16仅做固定6条枚举与恢复，不设置最优选择目标。
- R12/R13的默认benchmark loader当前实际使用CPU，即使task.device写cuda；回归记录真实device，硬件不足作为环境阻塞，不要求越权改runtime。
- 每题预算列于机器对象；这些数字是上限而非性能预估，也不声称在当前容器已经验证了运行时间。

## 5. 运行与交付顺序

1. 维护者检查本题机器定义、可信seed、源码基线和环境assets lock；预检返回可用后分派隔离candidate。准备阶段不执行20道题。
2. agent读取本卡和合同，写expected_code_diff与操作范围；通过题包record入口保留真实命令。对于已有缺陷先保存baseline行为再修改。
3. agent在允许路径实现本题，通过真实正例与公开负例，保留失败及新的attempt；使用规范experiment目录管理数据。
4. 汇集evidence、来源差异、运行日志与artifact hashes；候选状态保持unchecked。无产品改动的题明确交付空产品diff。
5. 独立reviewer在新的受控环境重放本题正反例，查看真实source、import路线、数值和副作用，而非只核对agent写的状态字段。
6. 汇总每题结果和失败归因。环境缺口、产品缺陷、规范缺口、执行错误分别报告，基于失败再由维护者更新后续版本的harness。考生不能改规则或评分来通过本轮。

实际命令参数以 [tools](../../tests/wp21/tools) 的--help及本题卡为准。candidate的证据目录固定为 `harness/tests/wp21/runs/<run_id>/<case_id>/`。产品源码只能修改机器清单中的allowed_paths；runtime、治理、fixtures、评分、历史结果和正式Tree保持受保护。

## 6. 独立验收与评分

| 维度 | 分数 | 实际证据 |
| --- | ---: | --- |
| 真实行为 | 40 | 真实checkpoint重放、原始向量/指标、文件副作用与case正例 |
| 兼容性和异常行为 | 25 | 默认行为、边界、负例、删除后的fresh process与回归 |
| 来源和运行证据 | 20 | 源码/配置/输入/checkpoint/环境身份、命令、退出码、hash及失败保留 |
| 范围和说明 | 15 | allowlist、目录规范、Expected/Observed、候选双语Context/Code |

完整PASS要求全部必需正反例独立复跑且真实性与范围硬门槛通过。结构audit成功仅进入EVIDENCE_READY_FOR_REPLAY，不自动判分。数值下降、模型行为改善或论文结论并非题目成功标准。训练和小样本比较只支持本题的工程结论。

## 7. 机器可读合同绑定

下面声明与cases.json同步，含5个always-on ID。标准registry parser可直接检查；case tags均来自现有catalog。

```harness-contracts-json
{
  "registry": "saebase.harness.contract_registry.v1",
  "plan": [
    "CT-ERROR"
  ],
  "cases": {
    "R01": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "data"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-NAMESPACE"
      ]
    },
    "R02": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-NAMESPACE"
      ]
    },
    "R03": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "data"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-NAMESPACE"
      ]
    },
    "R04": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-NAMESPACE"
      ]
    },
    "R05": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "recovery"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-RECOVERY",
        "CT-NAMESPACE"
      ]
    },
    "R06": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "data"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-NAMESPACE"
      ]
    },
    "R07": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "data",
        "recovery"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-RECOVERY",
        "CT-NAMESPACE"
      ]
    },
    "R08": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "data"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-NAMESPACE"
      ]
    },
    "R09": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "recovery"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-RECOVERY",
        "CT-NAMESPACE"
      ]
    },
    "R10": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "data"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-NAMESPACE"
      ]
    },
    "R11": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "recovery"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-RECOVERY",
        "CT-NAMESPACE"
      ]
    },
    "R12": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-NAMESPACE"
      ]
    },
    "R13": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "recovery"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-RECOVERY",
        "CT-NAMESPACE"
      ]
    },
    "R14": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "data"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-NAMESPACE"
      ]
    },
    "R15": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "data"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-NAMESPACE"
      ]
    },
    "R16": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "data",
        "recovery"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-RECOVERY",
        "CT-NAMESPACE"
      ]
    },
    "R17": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "data",
        "recovery"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-RECOVERY",
        "CT-NAMESPACE"
      ]
    },
    "R18": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "recovery"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-RECOVERY",
        "CT-NAMESPACE"
      ]
    },
    "R19": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "code",
        "tree",
        "context",
        "localization",
        "review",
        "recovery"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-CONTEXT",
        "CT-AST",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-RECOVERY",
        "CT-NAMESPACE"
      ]
    },
    "R20": {
      "tags": [
        "evidence",
        "environment",
        "resource",
        "config",
        "validation",
        "interface",
        "determinism",
        "namespace",
        "data",
        "recovery"
      ],
      "contracts": [
        "CT-BASE",
        "CT-SCOPE",
        "CT-TRUTH",
        "CT-CHECK",
        "CT-TRACE",
        "CT-ERROR",
        "CT-EVIDENCE",
        "CT-ENV",
        "CT-DETERMINISM",
        "CT-INTERFACE",
        "CT-DATA",
        "CT-CONFIG",
        "CT-RESOURCE",
        "CT-RECOVERY",
        "CT-NAMESPACE"
      ]
    }
  }
}
```

## 8. 本次交付边界

本包提供规范增补、完整题面、可信删除seed、分派/预检/命令记录/证据完整性工具与本地验证记录。它不声称20题已经由20个agent执行，也不声称真实checkpoint独立grader已经完成。后续跑题产生的结果归入每位agent自己的工作区，GitHub保持原状。

