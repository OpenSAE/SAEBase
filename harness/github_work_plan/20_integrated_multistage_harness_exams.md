# Work Plan 20 · 中大型多阶段集成考试与可信验收设计

状态：**SPEC_REVIEWED / DESIGN ONLY，2026-10-29**。本计划新增 X01–X20 共 20 道中大型任务，每道涉及至少三个阶段、多个实际模块边界、正反例、状态变化或变形关系，且已预先选定 Contract。**当前只完成任务设计和规范校验，并未实施 20 个独立行为 Oracle，不应把这些题标为 READY 或直接启动考试 Agent。**

承接 [WP18](18_harness_local_agent_examination_and_independent_acceptance.md) 与 [WP19](19_harness_large_scale_examination_and_verifier_assurance.md)，转向**真实行为、跨模块集成和长时程执行**。WP19 的 200 道小型合成考试作为已保存的历史基线，不重复计为 WP20 的完成成果。全部 X 系列题目以 `harness/contracts/wp20_cases.json` 为机器可读权威设计，以下内容便于人类和本地执行者理解；具体案例允许的数据范围、验证资源与产物清单以该 JSON 和本文件一致的定义为准。

## 0. 总目标与设计边界

WP19 的实践证明，Agent 可以很好地完成清晰的合成判断，但仅检查 `before == after`、`revision == expected` 或列表排序，并不能证明文件原子写入、多 Agent 合并或 AUROC 等真实行为。因此 WP20 不再用一个通用布尔算子冒充跨模块功能。**任务规模应体现在需要保持多步不变量，而不是样本数量或 GPU 消耗。**

每题至少包含：

1. 输入、状态或源码读取后进行真实工作，不仅验证一条字段存在性。
2. 两个以上实质上协作的模块或 API 边界，连续三至四个阶段。
3. 两种预先声明的合法情况、两种负面/故障注入、至少一条变形或状态不变量。
4. 明确可由独立参考计算、AST 比对、临时文件系统、事件重放或可信日志验证的真实**行为输出**。
5. 可溯源交付与可信 reviewer；模拟批准只能作为实验输入，永远不能代替人类对权威源代码的批准。

默认只采用 2–12 条 synthetic 样本或 2–12 个节点、CPU、无外网、无 GPU、无真实模型下载，单个场景的初始预算不超过 120 秒和 1024 MB。长时程是事件序列多，不是运行时间长。高成本扩展须另行预算和授权。

**保护规则**：只读参考 SAEBase 生产代码，考试 Agent 的允许写入路径限定在新建 `harness/tests/wp20/runs/<run_id>/<case_id>/` 下；禁止更改权威 Atlas、Contract Registry、WP20 规范、验证器、CI、Hooks、旧 Run 或生产模块。不可仅靠文件 Hook 作为系统级安全沙箱。候选代码由可信 reviewer 在可丢弃、无凭据、限权工作区执行。任务 Agent 无权写人工审批结果、独立评分和正式发布资产。

## 1. 二十题概览

| 题号 | 跨模块任务 | 阶段数 | 条件 Contract 数 | 当前状态 |
| --- | --- | ---: | ---: | --- |
| X01 | 真实源码变更到 Code Tree 与候选 Diff | 4 | 5 | SPEC_REVIEWED |
| X02 | Glossary 同步、双向链接与中英 Overlay | 4 | 5 | SPEC_REVIEWED |
| X03 | Atlas 五树切换与审阅会话状态恢复 | 4 | 5 | SPEC_REVIEWED |
| X04 | 可信命令日志、Receipt 与独立 PR 决策 | 4 | 5 | SPEC_REVIEWED |
| X05 | 跨工具读写授权与路径绕过检测 | 4 | 4 | SPEC_REVIEWED |
| X06 | 数据 Manifest 到分割、抽样与报告的完整流水线 | 4 | 4 | SPEC_REVIEWED |
| X07 | Model 与 SAE 身份校验、Activation 缓存与重放 | 4 | 5 | SPEC_REVIEWED |
| X08 | Tiny SAE 编码、解码、TopK 与批量等价性 | 4 | 4 | SPEC_REVIEWED |
| X09 | 多 Feature 干预链、撤销及生成模式隔离 | 4 | 4 | SPEC_REVIEWED |
| X10 | 特征证据、Token 位置与组件归因 | 4 | 4 | SPEC_REVIEWED |
| X11 | AUROC、AUPRC、Recall 的端到端指标引擎 | 4 | 4 | SPEC_REVIEWED |
| X12 | Bootstrap、负控制与区间估计重放 | 4 | 4 | SPEC_REVIEWED |
| X13 | Benchmark 适配器到汇总报告的多任务一致性 | 4 | 4 | SPEC_REVIEWED |
| X14 | Hydra 组合配置到 CPU Dry-run 与 Receipt | 4 | 5 | SPEC_REVIEWED |
| X15 | 断点中断、Checkpoint 校验与幂等恢复 | 4 | 5 | SPEC_REVIEWED |
| X16 | 安全归档、解包与原子提交 | 4 | 4 | SPEC_REVIEWED |
| X17 | 源码 SHA、CI 状态与人工审核的发布门禁 | 4 | 5 | SPEC_REVIEWED |
| X18 | 双 Agent 并发提交、冲突检测与安全合并 | 4 | 5 | SPEC_REVIEWED |
| X19 | 微型端到端 SAE 研究工作流 | 4 | 5 | SPEC_REVIEWED |
| X20 | 长时程交接、需求变更、中断恢复及终局审阅 | 4 | 6 | SPEC_REVIEWED |

四组各五题。正式运行时按 2–4 题小批次组织，优先从 X01、X06、X11、X16 这类具有明确参考行为的任务开始，先验证独立 grader 自身，再逐批让 Agent 执行。

## 2. 任务绑定与可信权属

沿用现有 22 个 Universal Contract，**无需新建 20 个 Contract**。Always-on 自动继承；每题另外按实际领域选择已存在的条件 Contract。以下为 Registry 可直接解析的声明。每个新题必须先校验绑定，再进入参考程序开发。

```harness-contracts-json
{
  "registry": "saebase.harness.contract_registry.v1",
  "plan": [
    "CT-ERROR"
  ],
  "cases": {
    "X01": {
      "tags": [
        "code",
        "tree",
        "evidence",
        "determinism",
        "interface"
      ],
      "contracts": [
        "CT-AST",
        "CT-STRUCT",
        "CT-EVIDENCE",
        "CT-DETERMINISM",
        "CT-INTERFACE"
      ]
    },
    "X02": {
      "tags": [
        "context",
        "evidence",
        "localization",
        "tree",
        "interface"
      ],
      "contracts": [
        "CT-CONTEXT",
        "CT-EVIDENCE",
        "CT-LOCALE",
        "CT-STRUCT",
        "CT-INTERFACE"
      ]
    },
    "X03": {
      "tags": [
        "tree",
        "review",
        "localization",
        "determinism",
        "interface"
      ],
      "contracts": [
        "CT-STRUCT",
        "CT-REVIEW",
        "CT-LOCALE",
        "CT-DETERMINISM",
        "CT-INTERFACE"
      ]
    },
    "X04": {
      "tags": [
        "evidence",
        "review",
        "privacy"
      ],
      "contracts": [
        "CT-TRACE",
        "CT-EVIDENCE",
        "CT-SCOPE",
        "CT-REVIEW",
        "CT-PRIVACY"
      ]
    },
    "X05": {
      "tags": [
        "privacy",
        "evidence",
        "interface"
      ],
      "contracts": [
        "CT-SCOPE",
        "CT-PRIVACY",
        "CT-EVIDENCE",
        "CT-INTERFACE"
      ]
    },
    "X06": {
      "tags": [
        "data",
        "evidence",
        "determinism",
        "interface"
      ],
      "contracts": [
        "CT-DATA",
        "CT-EVIDENCE",
        "CT-DETERMINISM",
        "CT-INTERFACE"
      ]
    },
    "X07": {
      "tags": [
        "data",
        "config",
        "evidence",
        "determinism",
        "interface"
      ],
      "contracts": [
        "CT-DATA",
        "CT-CONFIG",
        "CT-EVIDENCE",
        "CT-DETERMINISM",
        "CT-INTERFACE"
      ]
    },
    "X08": {
      "tags": [
        "data",
        "config",
        "determinism",
        "interface"
      ],
      "contracts": [
        "CT-DATA",
        "CT-CONFIG",
        "CT-DETERMINISM",
        "CT-INTERFACE"
      ]
    },
    "X09": {
      "tags": [
        "config",
        "data",
        "determinism",
        "interface"
      ],
      "contracts": [
        "CT-CONFIG",
        "CT-DATA",
        "CT-DETERMINISM",
        "CT-INTERFACE"
      ]
    },
    "X10": {
      "tags": [
        "data",
        "evidence",
        "determinism",
        "interface"
      ],
      "contracts": [
        "CT-DATA",
        "CT-EVIDENCE",
        "CT-DETERMINISM",
        "CT-INTERFACE"
      ]
    },
    "X11": {
      "tags": [
        "data",
        "config",
        "validation",
        "determinism"
      ],
      "contracts": [
        "CT-DATA",
        "CT-CONFIG",
        "CT-ERROR",
        "CT-DETERMINISM"
      ]
    },
    "X12": {
      "tags": [
        "data",
        "determinism",
        "evidence",
        "interface"
      ],
      "contracts": [
        "CT-DATA",
        "CT-DETERMINISM",
        "CT-EVIDENCE",
        "CT-INTERFACE"
      ]
    },
    "X13": {
      "tags": [
        "data",
        "determinism",
        "evidence",
        "interface"
      ],
      "contracts": [
        "CT-DATA",
        "CT-DETERMINISM",
        "CT-EVIDENCE",
        "CT-INTERFACE"
      ]
    },
    "X14": {
      "tags": [
        "config",
        "data",
        "determinism",
        "resource"
      ],
      "contracts": [
        "CT-CONFIG",
        "CT-DATA",
        "CT-DETERMINISM",
        "CT-TRACE",
        "CT-RESOURCE"
      ]
    },
    "X15": {
      "tags": [
        "recovery",
        "data",
        "evidence",
        "determinism",
        "interface"
      ],
      "contracts": [
        "CT-RECOVERY",
        "CT-DATA",
        "CT-EVIDENCE",
        "CT-DETERMINISM",
        "CT-INTERFACE"
      ]
    },
    "X16": {
      "tags": [
        "evidence",
        "recovery",
        "privacy"
      ],
      "contracts": [
        "CT-EVIDENCE",
        "CT-RECOVERY",
        "CT-SCOPE",
        "CT-PRIVACY"
      ]
    },
    "X17": {
      "tags": [
        "release",
        "evidence",
        "review",
        "determinism"
      ],
      "contracts": [
        "CT-RELEASE",
        "CT-EVIDENCE",
        "CT-REVIEW",
        "CT-TRACE",
        "CT-DETERMINISM"
      ]
    },
    "X18": {
      "tags": [
        "recovery",
        "review",
        "code"
      ],
      "contracts": [
        "CT-RECOVERY",
        "CT-REVIEW",
        "CT-TRACE",
        "CT-SCOPE",
        "CT-AST"
      ]
    },
    "X19": {
      "tags": [
        "data",
        "config",
        "evidence",
        "determinism",
        "interface"
      ],
      "contracts": [
        "CT-DATA",
        "CT-CONFIG",
        "CT-EVIDENCE",
        "CT-DETERMINISM",
        "CT-INTERFACE"
      ]
    },
    "X20": {
      "tags": [
        "review",
        "recovery",
        "evidence",
        "tree",
        "determinism"
      ],
      "contracts": [
        "CT-REVIEW",
        "CT-RECOVERY",
        "CT-TRACE",
        "CT-EVIDENCE",
        "CT-STRUCT",
        "CT-DETERMINISM"
      ]
    }
  }
}
```

**角色和状态**：`SPEC_REVIEWED → FROZEN → VERIFIER_QUALIFIED → READY → AGENT_SUBMITTED → INDEPENDENT_EVALUATED`。本轮全部 20 题停在 `SPEC_REVIEWED`。治理者确认规则完整并实现、校准 Reference 后才可以在新版本中升状态。Agent 不能私自把计划题目变成 READY。

独立验证器开发需提供：一份经确认的正确参考实现、至少两类明确不同的错误实现、合法场景误报控制、异常/恢复状态检查、基于源码或实际产物的证据核验。每个题目都要证明原来的目标确实被考到，例如 X11 要输出真正的 AUROC/AP/Recall 数值，而不是把排序或比例检查重新命名为这些指标。

## 3. Harness、Atlas 和真实证据链

### X01 · 真实源码变更到 Code Tree 与候选 Diff

**要完成的实际功能**：以隔离 synthetic Python 模块的两次修改为输入，生成真实 AST 身份、Tree 更新和可审阅的来源证据。

**关联仓库接口（只读作为设计依据）**：`harness/code/build.py`、`harness/code/diff.py`、`harness/code/interfaces.py`、`harness/core/agent_proposal_review.py`。

**执行过程**

1. 读取基线源码并建立确定性 AST 接口表
2. 应用一次函数改名与一次参数变化并重新构建
3. 生成结构 Diff、来源行号与 unchecked 候选 Tree
4. 核对 Expected/Observed 与哈希、保留无法静态证明的 calls

**必须通过的两组正常场景**：（1）相同函数只调整空行时稳定身份与职责不变；（2）函数改名加默认参数变化时旧 ID、新 ID、签名差异均有来源。

**必须识别的两组异常**：（1）源码真实函数被遗漏但候选仍报告 complete；（2）虚构动态调用边或在无来源时标 verified。

**变形或状态验证**：源码只是重排行号而不改变语义时，能力集合及稳定映射应保持一致，行号证据必须更新。

**独立参考依据**：用 Python ast 从两份可信 fixture 重新抽取函数、签名与父边；比较集合与节点路径；人工抽查语义标题。

**交付的关键产物**：`before_ast.json`、`after_ast.json`、`observed_diff.json`、`review_candidate.json`、`provenance.json`。

**条件 Contract**：`CT-AST`、`CT-STRUCT`、`CT-EVIDENCE`、`CT-DETERMINISM`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X02 · Glossary 同步、双向链接与中英 Overlay

**要完成的实际功能**：对包含六个节点的知识结构执行术语新增、修改、双向映射和中英本地化同步。

**关联仓库接口（只读作为设计依据）**：`harness/knowledge/terms.py`、`harness/knowledge/validate.py`、`harness/visualization/build_language.py`、`harness/visualization/language.py`。

**执行过程**

1. 载入节点、canonical term 和关联证据
2. 加入新术语并修正两处双向关系
3. 构造 zh/en Overlay，保持代码符号与源 ID
4. 输出冲突、缺失翻译与受影响节点清单

**必须通过的两组正常场景**：（1）新增 term 关联两个真实节点，正向和反向边完全一致；（2）修改一个术语描述只改变实际依赖它的 Knowledge 节点。

**必须识别的两组异常**：（1）term 缺失或重复仍被标记有效；（2）English translation unavailable 或中文占位被视为完整翻译。

**变形或状态验证**：调整无关节点的展示顺序，不改变术语 ID、双向关系和翻译覆盖率分母。

**独立参考依据**：从固定术语图独立重建反向索引，检查覆盖率和双语字段，保留语义争议以供人审。

**交付的关键产物**：`terms.json`、`nodes.json`、`locale_zh.json`、`locale_en.json`、`impact_report.json`。

**条件 Contract**：`CT-CONTEXT`、`CT-EVIDENCE`、`CT-LOCALE`、`CT-STRUCT`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X03 · Atlas 五树切换与审阅会话状态恢复

**要完成的实际功能**：在独立的五树小图上完成语言切换、节点聚焦、审核、撤销和重新加载。

**关联仓库接口（只读作为设计依据）**：`harness/visualization/app.js`、`harness/visualization/atlas-structure.js`、`harness/core/agent_proposal_review.py`。

**执行过程**

1. 为 Core/Knowledge/Code/Environment/Context 初始化独立焦点与展开状态
2. 连续执行树切换及中英切换并保留每棵树状态
3. 执行 unchecked→deny→undo 的可信草稿事件
4. 序列化状态并在新进程恢复后重放验证

**必须通过的两组正常场景**：（1）在 Code 中选中节点后切到 Core 再返回，Code 的焦点及展开集合不变；（2）撤销仅复位本 Tree 的目标节点且 review revision 递增。

**必须识别的两组异常**：（1）不存在的节点/Tree 被静默转换成空队列；（2）一个 Tree 的批准结果感染另一 Tree 的同名节点。

**变形或状态验证**：改变 UI 语言时，所有节点 ID、焦点、审核状态和图关系应完全保持。

**独立参考依据**：以治理方状态机参考重新执行事件日志，逐事件比较状态；任何正式 Approve 只能由模拟受信任 reviewer 签发。

**交付的关键产物**：`event_log.json`、`state_before.json`、`state_after.json`、`replay.json`。

**条件 Contract**：`CT-STRUCT`、`CT-REVIEW`、`CT-LOCALE`、`CT-DETERMINISM`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X04 · 可信命令日志、Receipt 与独立 PR 决策

**要完成的实际功能**：将合成命令执行日志、文件 Diff、源码哈希和 Contract Receipt 组合成一次不可伪造的技术验收。

**关联仓库接口（只读作为设计依据）**：`harness/contracts/trace.py`、`harness/contracts/acceptance.py`、`harness/contracts/compliance.py`、`harness/core/agent_flow.py`。

**执行过程**

1. 解析工作单、适用 Contract 与独立运行身份
2. 收集真实模拟 runner 日志及 Git diff
3. 对 Receipt、输出路径、来源 SHA 和命令退出码逐项对账
4. 给出技术状态及人工审批仍需独立完成的证据

**必须通过的两组正常场景**：（1）成功案例所有声明变更与观察日志完全一致；（2）某命令失败后保留原退出码及可恢复的新 Run 记录。

**必须识别的两组异常**：（1）没有真实执行就通过自行生成 exit_code=0 获得通过；（2）selfcheck PASS 被提升成 human_approved。

**变形或状态验证**：重新排列相互独立的日志行不应改变覆盖关系，但必须保留真实时间与因果证据。

**独立参考依据**：独立 runner 产生受保护的退出码/文件集合；对比 Receipt、SHA 与真实变更；禁止从候选的 PASS.json 取结论。

**交付的关键产物**：`receipt.json`、`command_trace.json`、`git_diff.json`、`technical_verdict.json`。

**条件 Contract**：`CT-TRACE`、`CT-EVIDENCE`、`CT-SCOPE`、`CT-REVIEW`、`CT-PRIVACY`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X05 · 跨工具读写授权与路径绕过检测

**要完成的实际功能**：在临时目录模拟 Agent 使用文件 API、Shell、重定向、重命名和子代理传递工作区权限。

**关联仓库接口（只读作为设计依据）**：`.cursor/hooks/guard_paths.py`、`.cursor/hooks/guard_shell.py`、`harness/contracts/selection.py`、`harness/contracts/compliance.py`。

**执行过程**

1. 冻结仅允许写入的工作区根与读取列表
2. 分别用 API、shell 和 rename 执行合法同源写入
3. 注入 ../、绝对路径、软链接、重定向和委托越权
4. 生成被拒绝请求及实际文件系统的无副作用证明

**必须通过的两组正常场景**：（1）工作区中的正常文件新建、读取与原子修改成功；（2）合法的只读 Contract 查询成功，结果不会提升为写权限。

**必须识别的两组异常**：（1）通过符号链接或间接命令写进治理文件；（2）子代理将受限权限升为可修改独立评分器。

**变形或状态验证**：等价合法相对路径经规范化保持授权决定；逃逸路径无论由哪种操作提交均须拒绝。

**独立参考依据**：用独立的只读文件快照+OS 限权临时目录对比副作用；仅模拟攻击数据，严禁对真实仓库执行破坏测试。

**交付的关键产物**：`attempts.json`、`denials.json`、`filesystem_before.json`、`filesystem_after.json`。

**条件 Contract**：`CT-SCOPE`、`CT-PRIVACY`、`CT-EVIDENCE`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。


## 4. 数据、SAE 与特征操作

### X06 · 数据 Manifest 到分割、抽样与报告的完整流水线

**要完成的实际功能**：从小型合成 prompt 数据集建立身份、哈希、train/val/test 分割和可重复的测试报告。

**关联仓库接口（只读作为设计依据）**：`saebase/data/prompt_schema.py`、`saebase/data/prompt_store.py`、`saebase/data/artifact_layout.py`、`saebase/benchmarks/schema.py`。

**执行过程**

1. 读取 8 个稳定样本、prompt 与 label 的组合
2. 验证 Manifest 并生成无交集分割
3. 选择少量目标样本运行 CPU 假模型评分
4. 报告实际覆盖数、样本身份与结果溯源

**必须通过的两组正常场景**：（1）8 个样本无损划分，train/val/test 计数守恒；（2）更换输入顺序后，按 ID 规范化的分割和汇总不变。

**必须识别的两组异常**：（1）相同 sample_id 在 train 和 test 同时出现；（2）只评估 3 个样本却报告全量 8 个。

**变形或状态验证**：重排原文件和批次，不改变基于样本 ID 的结果、Manifest SHA 与覆盖率。

**独立参考依据**：独立 reference 以样本 ID 集合重建分割、逐项哈希及汇总；计数差异/漏报为硬失败。

**交付的关键产物**：`source_manifest.json`、`splits.json`、`predictions.json`、`coverage_report.json`。

**条件 Contract**：`CT-DATA`、`CT-EVIDENCE`、`CT-DETERMINISM`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X07 · Model 与 SAE 身份校验、Activation 缓存与重放

**要完成的实际功能**：在 tiny CPU 模型替身上保存激活、SAE 信息、token 对齐及可重放缓存。

**关联仓库接口（只读作为设计依据）**：`saebase/sae/loading.py`、`saebase/sae/contracts.py`、`saebase/store/identity.py`、`saebase/store/shards.py`。

**执行过程**

1. 建立 model_id、SAE ID、hook 层、tokenizer 与 checkpoint 的显式身份
2. 产生两批 tiny Activation 并规范化样本和 token ID
3. 按身份写入缓存与 Manifest
4. 从缓存重放并对比原生生成激活的形状与数值

**必须通过的两组正常场景**：（1）同一身份、seed、输入重放得到逐 token 一致的 Activation；（2）两批顺序重排后以 sample ID 恢复与原顺序对应的结果。

**必须识别的两组异常**：（1）换 model SHA 或 layer 仍错误命中缓存；（2）SAE d_in 与 model hidden 不符却继续读取。

**变形或状态验证**：改变 batch 划分但保持输入身份时，规范化后的激活数值及模型标识相同。

**独立参考依据**：使用独立 NumPy 小数组构造确定性 Activation；重算缓存 SHA、逐 token 比对；无模型权重下载。

**交付的关键产物**：`identities.json`、`activation_shards.json`、`cache_manifest.json`、`replay_diff.json`。

**条件 Contract**：`CT-DATA`、`CT-CONFIG`、`CT-EVIDENCE`、`CT-DETERMINISM`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X08 · Tiny SAE 编码、解码、TopK 与批量等价性

**要完成的实际功能**：使用固定小矩阵实现批处理 SAE 编码解码，检查稀疏约束、TopK 并列和数值误差。

**关联仓库接口（只读作为设计依据）**：`saebase/sae/sparsify_adapter.py`、`saebase/sae/task.py`、`saebase/benchmarks/metrics`。

**执行过程**

1. 定义 d_in=3、d_sae=5 的小型编码器和解码器矩阵
2. 对 4 个小输入计算 dense 与 TopK 稀疏激活
3. 解码还原并计算每样本重构误差
4. 跨不同 batch 大小验证逐样本数值和特征 ID 稳定

**必须通过的两组正常场景**：（1）固定矩阵输入手算 1 个编码与解码结果符合容差；（2）相同分数并列时按照预定义 feature ID 次序选择 TopK。

**必须识别的两组异常**：（1）TopK 允许超过 feature count 或负 K；（2）只更改 batch size 就改变同一输入的激活或误差。

**变形或状态验证**：样本顺序重排后按 ID 对齐的结果完全一致；k=d_sae 与全保留的编码一致。

**独立参考依据**：独立矩阵乘法 reference（NumPy 或手算）逐元素比较，检查 TopK identity 和完整分母。

**交付的关键产物**：`tiny_weights.json`、`encoded.json`、`decoded.json`、`reconstruction.json`。

**条件 Contract**：`CT-DATA`、`CT-CONFIG`、`CT-DETERMINISM`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X09 · 多 Feature 干预链、撤销及生成模式隔离

**要完成的实际功能**：执行 Ablation、Scale、Add、Clamp 的有序组合，验证回滚、共享状态清理和 teacher-forced/generation 的明确区分。

**关联仓库接口（只读作为设计依据）**：`saebase/interventions/operators.py`、`saebase/interventions/engine.py`、`saebase/interventions/specs.py`、`saebase/interventions/results.py`。

**执行过程**

1. 创建含负值和零激活的 5 维 tiny 特征向量
2. 在 token/layer 上按给定次序执行四种干预
3. 保存每步 before/after 的特征集合和最终数值
4. 撤销所有干预并在不同请求模式重放以检查无残留

**必须通过的两组正常场景**：（1）只在目标 token 和 feature 上改变数值，其他位置 bitwise 相同；（2）取消干预后 baseline 恢复，下一请求不继承任何 hook。

**必须识别的两组异常**：（1）未知 feature ID 被接受且污染其他特征；（2）Scale→Add 与 Add→Scale 被错误视为可交换。

**变形或状态验证**：对无操作 scale=1、add=0 的输入，生成结果应与 baseline 完全一致。

**独立参考依据**：独立逐步数组 reference 比对每个中间状态及 rollback；禁止把整段效果简化成单个布尔判断。

**交付的关键产物**：`steps.json`、`before_after.json`、`rollback.json`、`mode_audit.json`。

**条件 Contract**：`CT-CONFIG`、`CT-DATA`、`CT-DETERMINISM`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X10 · 特征证据、Token 位置与组件归因

**要完成的实际功能**：从 tiny activation/feature trace 生成可追溯的 Top Feature 解释表及跨组件关联。

**关联仓库接口（只读作为设计依据）**：`saebase/api/features.py`、`saebase/api/component_analysis.py`、`saebase/sae/coverage.py`、`saebase/store/collect.py`。

**执行过程**

1. 构造 6 个 token × 4 个 feature 的合成激活表
2. 按指定 token 范围和 feature ID 提取证据
3. 生成 Top Feature 排名与来源 token 对应关系
4. 对不存在位置或来源缺口输出明确 unresolved

**必须通过的两组正常场景**：（1）相同激活分数的 feature 排序稳定且来源 token 正确；（2）只改变无关 feature 不影响目标 feature 的来源链。

**必须识别的两组异常**：（1）缺失 token 被补成有证据的特征；（2）top feature 标注来源位置与原数据索引不一致。

**变形或状态验证**：输入的无关 token 被插入时，按稳定 token ID 选择的证据必须保持。

**独立参考依据**：独立从原始激活矩阵计算排名与 token 映射；任何没有原始记录支持的解释标 unresolved。

**交付的关键产物**：`trace.json`、`feature_table.json`、`token_evidence.json`、`unresolved.json`。

**条件 Contract**：`CT-DATA`、`CT-EVIDENCE`、`CT-DETERMINISM`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。


## 5. 指标、实验与可复现性

### X11 · AUROC、AUPRC、Recall 的端到端指标引擎

**要完成的实际功能**：对真实标签和预测分数计算三项指标，明确并列分数、缺失类别和阈值规则。

**关联仓库接口（只读作为设计依据）**：`saebase/benchmarks/metrics`、`saebase/benchmarks/report.py`、`saebase/benchmarks/schema.py`。

**执行过程**

1. 载入含正负样本及重复分数的二分类评估集
2. 计算完整 AUROC、Average Precision 口径 AUPRC 和阈值 Recall
3. 报告每个指标所用分母、阈值、样本 ID
4. 对零正类等未定义指标进行显式标记与来源记录

**必须通过的两组正常场景**：（1）标签 [1,0]、分数 [0.9,0.1] 对应 AUROC=1、AP=1；（2）标签 [1,0]、分数 [0.1,0.9] 对应 AUROC=0、AP=0.5。

**必须识别的两组异常**：（1）只检查输入分数是否排序便返回所谓 AUROC；（2）没有正类仍无说明地报告 AUPRC=1 或 Recall=0。

**变形或状态验证**：保持标签与分数配对后整体重排，AUROC/AP/Recall 均不变；单调递增分数变换不改变 AUROC。

**独立参考依据**：用 pairwise ranking 手算 AUROC，并用 independent PR stair-step average precision 计算 AP；比较参考库须固定积分口径。

**交付的关键产物**：`observations.json`、`metrics.json`、`thresholds.json`、`undefined_cases.json`。

**条件 Contract**：`CT-DATA`、`CT-CONFIG`、`CT-ERROR`、`CT-DETERMINISM`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X12 · Bootstrap、负控制与区间估计重放

**要完成的实际功能**：在少量成组样本中执行固定 seed 的 Bootstrap 和一种明确负控制，输出可重放的不确定性报告。

**关联仓库接口（只读作为设计依据）**：`saebase/reliability/bootstrap.py`、`saebase/reliability/resampling.py`、`saebase/reliability/controls.py`、`saebase/reliability/report.py`。

**执行过程**

1. 选定 6 个分组观测与明确重采样单位
2. 使用固定 seed 重抽样 20 次生成统计量
3. 应用打乱标签的负控制并保持样本组关系
4. 输出点估计、分位数区间、seed 和每轮索引

**必须通过的两组正常场景**：（1）固定 seed 和固定输入两次执行产生相同索引序列与置信区间；（2）按组抽样时完整同组观测一同出现，不混用个体独立抽样。

**必须识别的两组异常**：（1）在样本数量为零时伪造合法置信区间；（2）对负控制重用原始未打乱数据却宣称打乱。

**变形或状态验证**：在固定抽样索引下改变无关记录顺序，统计量保持一致；seed 变化需记录并可重放。

**独立参考依据**：治理方提供固定 indices 的独立 resampling reference；比较每次抽样 ID、分位数和负控制有效性。

**交付的关键产物**：`resample_indices.json`、`replicates.json`、`controls.json`、`intervals.json`。

**条件 Contract**：`CT-DATA`、`CT-DETERMINISM`、`CT-EVIDENCE`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X13 · Benchmark 适配器到汇总报告的多任务一致性

**要完成的实际功能**：汇合两个 tiny Benchmark Adapter 的不同输出，统一样本权重、失败状态与指标报告。

**关联仓库接口（只读作为设计依据）**：`saebase/benchmarks/adapters`、`saebase/benchmarks/runner.py`、`saebase/benchmarks/registry.py`、`saebase/benchmarks/report.py`。

**执行过程**

1. 定义两个具有不同指标和样本数量的合成 adapter
2. 调用 registry 与 planner 执行 CPU tiny evaluations
3. 标准化结果、失败/缺失和分母信息
4. 聚合跨 adapter 报告并核对来源哈希

**必须通过的两组正常场景**：（1）样本数 2 和 6 的均值按样本数而不是简单平均批均值；（2）一个 adapter 失败时其缺失指标不被填 0 也不改另一 adapter。

**必须识别的两组异常**：（1）Adapter 输出重复 sample ID 被默默合并；（2）报告数值无法追溯回原始每样本结果。

**变形或状态验证**：改变 adapter 执行顺序不应改变统计结果或输出 ID。

**独立参考依据**：独立重算每样本指标加权汇总；不同 adapter 的错误状态与来源证据逐条校验。

**交付的关键产物**：`adapter_results.json`、`normalized_results.json`、`aggregate_report.json`。

**条件 Contract**：`CT-DATA`、`CT-DETERMINISM`、`CT-EVIDENCE`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X14 · Hydra 组合配置到 CPU Dry-run 与 Receipt

**要完成的实际功能**：从多组模型、数据与任务配置生成少量允许的运行计划，验证覆盖优先级与最终执行证据。

**关联仓库接口（只读作为设计依据）**：`saebase/runtime/train.py`、`saebase/runtime/experiment_paths.py`、`saebase/runner.py`、`saebase/registry.py`。

**执行过程**

1. 加载两个小型配置组并解析覆盖顺序
2. 在明确允许矩阵内生成至多 4 个 dry-run job
3. 为每个 job 计算稳定配置身份、预算与输出目录
4. 模拟执行与失败，收集最终 resolved config 和 Receipt

**必须通过的两组正常场景**：（1）同一 resolved config 与 seed 产生稳定配置摘要和路径；（2）覆盖 config 改变后仅受影响 job SHA 发生改变。

**必须识别的两组异常**：（1）未知 task 或 model 与 dataset 不兼容时仍启动运行；（2）Receipt 记录的配置与实际 resolved 配置不同。

**变形或状态验证**：重排同一配置 dict 的键序，不改变 canonical JSON 或配置身份。

**独立参考依据**：独立合成 config 的规范化 reference，比较 dry-run job 及 Receipt，不加载真实模型。

**交付的关键产物**：`resolved_config.json`、`jobs.json`、`receipts.json`、`execution_report.json`。

**条件 Contract**：`CT-CONFIG`、`CT-DATA`、`CT-DETERMINISM`、`CT-TRACE`、`CT-RESOURCE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X15 · 断点中断、Checkpoint 校验与幂等恢复

**要完成的实际功能**：以三步可中断 CPU pipeline 模拟运行崩溃、恢复和重复执行，保证结果一致与历史不可变。

**关联仓库接口（只读作为设计依据）**：`saebase/runtime/experiment_paths.py`、`saebase/runtime/summary_log.py`、`saebase/store/shards.py`、`saebase/reliability/runner.py`。

**执行过程**

1. 生成具有 run/config/data/seed 身份的输入和 checkpoint
2. 完成第一阶段后触发模拟异常并持久化部分结果
3. 校验 checkpoint 的身份后仅继续剩余阶段
4. 重复恢复并对比最终产物与一次成功运行的等价性

**必须通过的两组正常场景**：（1）同一身份恢复后结果与不中断运行一致，失败记录留存；（2）对同一 run 重复恢复不重复写结果、旧记录 SHA 不变。

**必须识别的两组异常**：（1）不同数据 SHA 的 checkpoint 被接受恢复；（2）中断时留下被标记 completed 的半写入结果。

**变形或状态验证**：调整中断点但保持数据身份、种子和实际步骤，最终结果与不间断执行等价。

**独立参考依据**：独立状态机对照每一步、文件哈希和重试幂等性；按故障注入点重放。

**交付的关键产物**：`checkpoint.json`、`events.json`、`final_artifacts.json`、`resume_audit.json`。

**条件 Contract**：`CT-RECOVERY`、`CT-DATA`、`CT-EVIDENCE`、`CT-DETERMINISM`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。


## 6. 发布、并发与长时程协作

### X16 · 安全归档、解包与原子提交

**要完成的实际功能**：在临时文件系统中生成归档，注入损坏和越界成员，验证任何失败不得污染正式目录。

**关联仓库接口（只读作为设计依据）**：`saebase/tools/stage_d_archive.py`、`saebase/release/gates.py`、`saebase/tools/verify_distribution.py`。

**执行过程**

1. 构建三个 tiny 产物及完整 Manifest
2. 在受保护临时目录创建归档并验 SHA
3. 模拟压缩过程与解包阶段的故障/非法条目
4. 通过验证后原子提交，失败则恢复原始快照

**必须通过的两组正常场景**：（1）正常归档独立解包后字节 SHA 与原始文件完全相等；（2）模拟写入失败后正式目录字节不变且临时文件清理。

**必须识别的两组异常**：（1）tar 条目 '../secret' 通过归档解包逃逸；（2）归档损坏时仍将阶段标记 published。

**变形或状态验证**：改变归档成员顺序不改变解包后的规范化 Manifest 与文件字节集合。

**独立参考依据**：治理方在 disposable temp dir 上注入失败，比较逐字节快照与路径集合，永不操作真实发布目录。

**交付的关键产物**：`archive_manifest.json`、`extract_report.json`、`fault_trace.json`、`filesystem_snapshot.json`。

**条件 Contract**：`CT-EVIDENCE`、`CT-RECOVERY`、`CT-SCOPE`、`CT-PRIVACY`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X17 · 源码 SHA、CI 状态与人工审核的发布门禁

**要完成的实际功能**：模拟多条 CI 检查的依赖、不同源码 SHA、人工批准状态及变更后重新评估发布资格。

**关联仓库接口（只读作为设计依据）**：`saebase/release/gates.py`、`saebase/release/planner.py`、`saebase/release/runner.py`、`.github/workflows/ci.yml`。

**执行过程**

1. 构建 required-check DAG 与待发布源码身份
2. 插入 success、failure、skipped、pending 以及旧 SHA 的状态
3. 添加可信 reviewer 的模拟批准收据
4. 重新计算可发布决议并检查不可变 tag 与完整审计

**必须通过的两组正常场景**：（1）所有 required checks 在当前 SHA success 且受信任批准时可产生模拟 RELEASE_READY；（2）新提交产生后，之前的成功检查不允许继续用于新 SHA。

**必须识别的两组异常**：（1）skipped/pending 被错误视为 success；（2）自称 approved 的 Agent JSON 使真实 release 被执行。

**变形或状态验证**：添加非 required 且成功的检查不会使缺失的 required check 获得通过。

**独立参考依据**：治理方按当前 SHA 和 required-check ID 集合计算发布状态；不允许实际 GitHub 发布、tag 或外网。

**交付的关键产物**：`check_graph.json`、`approvals.json`、`release_decision.json`、`audit.json`。

**条件 Contract**：`CT-RELEASE`、`CT-EVIDENCE`、`CT-REVIEW`、`CT-TRACE`、`CT-DETERMINISM`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X18 · 双 Agent 并发提交、冲突检测与安全合并

**要完成的实际功能**：在临时 Git 仓库使用两个模拟执行者同时修改同一函数，重放提交顺序与显式冲突处理。

**关联仓库接口（只读作为设计依据）**：`harness/core/agent_flow.py`、`harness/core/agent_apply.py`、`harness/code/diff.py`、`harness/contracts/trace.py`。

**执行过程**

1. 创建 2 文件的 tiny Git 基线与两个 Agent 的独立授权
2. Agent A、B 从相同 SHA 修改有重叠的函数区域
3. A 提交后 B 在 expected SHA 失败并获取新源码
4. B 依据允许的变更重新应用、解决冲突并生成新验收记录

**必须通过的两组正常场景**：（1）不冲突的两处修改最终都可保留，且 history 包含双方来源；（2）真正重叠的冲突必须进入明确 RESOLUTION_REQUIRED 而不是直接覆盖。

**必须识别的两组异常**：（1）B 使用旧 SHA 直接 force 更新分支；（2）双方 Agent 的自评互相当作人类审核。

**变形或状态验证**：交换 A/B 执行先后，对无冲突的修改得到语义等价结果；存在冲突时保留双方提案记录。

**独立参考依据**：独立临时 Git repo 构造 expected SHA CAS，重算最终 AST 与作者归属，查强推及历史覆盖。

**交付的关键产物**：`base_commit.json`、`agent_events.json`、`merge_result.json`、`conflict_report.json`。

**条件 Contract**：`CT-RECOVERY`、`CT-REVIEW`、`CT-TRACE`、`CT-SCOPE`、`CT-AST`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X19 · 微型端到端 SAE 研究工作流

**要完成的实际功能**：将数据、激活、tiny SAE 编码、Feature 干预、指标评估与报告联系成一条真实 CPU 合成流水线。

**关联仓库接口（只读作为设计依据）**：`saebase/data/prompt_schema.py`、`saebase/sae/task.py`、`saebase/interventions/engine.py`、`saebase/benchmarks/runner.py`、`saebase/results.py`。

**执行过程**

1. 生成含 4 条样本的 synthetic 数据与确定性的 tiny model activation
2. 进行 SAE 编码和 TopK，再保存真实 feature ID
3. 对一个目标 feature 执行 Ablation 并重算预测输出
4. 计算 before/after 指标、来源与干预差异并形成报告

**必须通过的两组正常场景**：（1）无操作干预后输出与 baseline 完全相同；（2）Ablation 对至少一个特征产生可核验的非零预测差异且非目标 token 保持。

**必须识别的两组异常**：（1）报告说发生干预但实际特征和输出均未改变；（2）指标分母与有效样本集不一致或来源身份丢失。

**变形或状态验证**：batch 1 与 batch 4 的结果在稳定 sample ID 对齐后相同；同一 seed 可重现完整证据链。

**独立参考依据**：独立 tiny 数学 pipeline 逐阶段比较编码、干预与指标；误差/身份必须守恒，不接受截图作为数值证明。

**交付的关键产物**：`dataset.json`、`activation.json`、`encoded.json`、`interventions.json`、`metrics.json`、`provenance.json`。

**条件 Contract**：`CT-DATA`、`CT-CONFIG`、`CT-EVIDENCE`、`CT-DETERMINISM`、`CT-INTERFACE`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。

### X20 · 长时程交接、需求变更、中断恢复及终局审阅

**要完成的实际功能**：以 8–10 个预先定义的事件组成完整 workflow，包含多 Agent 交接、需求修订、两次失败注入和最后的候选 Tree 审阅。

**关联仓库接口（只读作为设计依据）**：`harness/core/agent_flow.py`、`harness/core/agent_proposal_review.py`、`harness/contracts/acceptance.py`、`harness/core/github_review.py`。

**执行过程**

1. A 领取受限工作单并完成第一段任务及可信记录
2. 治理方发布带 revision 的小需求变化，B 依据新工作单接手
3. 制造一次提交冲突与一次模拟中断，执行恢复与受限重试
4. 生成多阶段 Trace、Expected/Observed Diff、候选 Tree 并进行可信技术审查

**必须通过的两组正常场景**：（1）需求更新后保留先前有效不变量且 B 不继承 A 的扩大权限；（2）恢复后的最终 Artifact 与标准事件轨迹等价，历史失败不被抹去。

**必须识别的两组异常**：（1）旧审批继承到新源码 SHA 或其他 Agent；（2）最终全部测试绿但任一阶段出现未授权写仍然被判 PASS。

**变形或状态验证**：无冲突的两个执行者次序对换，不影响最终 Artifact；任何额外状态变化都必须进入事件日志。

**独立参考依据**：独立 reviewer 用固定事件机重放每个状态；对原始 Git diff、权限日志、所有中间状态逐步验收，而不只看最终文件。

**交付的关键产物**：`work_order.json`、`events.json`、`checkpoint.json`、`final_diff.json`、`candidate_tree.json`、`verdict_inputs.json`。

**条件 Contract**：`CT-REVIEW`、`CT-RECOVERY`、`CT-TRACE`、`CT-EVIDENCE`、`CT-STRUCT`、`CT-DETERMINISM`。状态为 `SPEC_REVIEWED`；独立 Oracle 尚需实现和校准。


## 7. WP20 统一运行、收据和候选 Tree 要求

每个 Case 的 **具体输入输出接口依据其实际任务**。多阶段 pipeline 不强制全部压缩到 `solve(payload)`；执行 Agent 应提供一个薄的可调用 orchestrator 和多个有实际职责的模块。测试应直接操作 tiny 数据、临时文件、状态机、真实 AST 或 Mock API。不同 Case 的核心行为不得全部归并成相同的布尔条件。

交付目录模板：

```text
harness/tests/wp20/runs/<new_run_id>/X01/
  src/                        # 可调用执行模块和相关辅助模块
  tests/                      # 功能测试、故障注入和变形测试
  fixtures/                   # 最小 synthetic 输入及期待状态
  output_manifest.json        # 全部实际产物与 SHA
  contract_receipt.json       # 适用 CT、真实命令、退出码、不可用项
  run_trace.json              # 逐阶段事件及观测数据
  review_candidate.json       # 实际源码 AST 对应的 unchecked 节点
  report.md                   # Expected / Observed / BLOCKED / FAIL
```

WP19 在一次性收集所有同名 `test_implementation.py` 时曾出现 pytest import mismatch；WP20 的不同 Case 统一使用互不冲突的 package/测试模块命名（或隔离子进程逐题测试），并为全套运行提供明确的 test discovery 机制。**不能为了绕过测试收集失败而删除测试或略过失败案例。**

工作区内可以让 Agent 使用受控公开的 verifier。但官方 verifier 必须由独立治理基线加载，产生的独立日志与候选代码分离。禁止把 Agent 的自测 PASS 或报告 SHA 当作独立技术验收。失败归因沿用 WP18/WP19 原有术语，失败记录永久保存，不随修复被改写。

## 8. 先实现验收，再启动 Agent

治理维护者后续执行顺序：

1. 为每题建立**独立的受保护 Oracle**，固定输入、预期输出、失败注入及来自真实仓库接口的可信行为定义。需要时复用已有参考，但确认该参考真的覆盖目标功能。
2. 使用独立 Good/Bad implementations 校准 grader。对 X11 指标、X16 文件原子性、X18 Git 并发等题，验证器必须执行真实数学、文件或 Git 操作；不能只比较一个状态标志。
3. 为各 Case 提供小型可运行的 scenario fixture 和不同内容的 reviewer 专用输入，记录固定 seed 与版本。检查 task 所用的 Contract 标签到实际行为一致。
4. 在 CI 中运行规范完整性、参考正确性、已知错误必须被发现、合法变化误报、scope gate 与资源限制。受信任 `main` 的 grader 决定行为结果，考生工作区只能提交候选。
5. 仅对完成以上条件的 Case 标记 `READY`，按小批次开放给本地 Agent。最终人工审阅、正式 Atlas 更新或产品发布必须另行批准。

**本轮 Definition of Done**：20 道有不同真实目标的任务、20 个预声明 Contract 绑定、模块依据、完整操作链、正反例、变形检查、独立 Oracle 设计和资源边界已经提交并经过规范一致性检查。它不是“20 题全部已经可以自动评分”的宣称。

**下一阶段的评估目标**：能否让同一个 Agent 在连续的来源读取、代码生成、执行、失败恢复及证据审查中，持续遵守跨步骤 Contract。通过率、验证器错误检出能力与历史回归应分开记录。
