# SAEBase Atlas 五树生成、编写与发布规范

> 本文档是 SAEBase Atlas 的**先验设计合同**。任何 Tree、Context、双语数据、可视化和网站部署修改，都先核对本文；生成器和验收脚本必须落实下述可执行准则。本文只约束 SAEBase，OpenResearch 保持独立的项目权威数据与部署。
>
> 规范版本：1.0 · 项目身份：`SAEBase` · 发布根：`/opensae/saebase/atlas/` · 权威源码：`OpenSAE/SAEBase`。

## 1. 更新目的与职责边界

Atlas 让研究者通过五棵结构化树，独立理解研究目标、科学知识、真实代码、执行环境和专业术语；所有表达必须能追溯到源码或经核验的材料。科研工作使用 Atlas 提供背景与定位，不把 UI 内容当作实验证据。

| Tree | 唯一回答的问题 | SAEBase 权威来源 | 可展示的实体 |
| --- | --- | --- | --- |
| Core | 为什么这样组织科研项目？ | `harness/core/project_graph.json` | 设计问题、方案、目标、管理约束 |
| Knowledge | 研究什么，方法依据是什么？ | `doc/knowledge/context_graph.json` | 研究问题、方案、可验证目标 |
| Code | 当前项目的真实实现是什么？ | Python AST 与 `doc/code/code_tree.json` | 文件、类、函数、方法、配置关联 |
| Environment | 实际在哪种条件下执行？ | `harness/environment/probe.py` 的实测快照及明确标记的合成示例 | 硬件、软件、资源与执行限制 |
| Context | 技术术语与科学概念是什么？ | `doc/knowledge/glossary.json`、`doc/knowledge/atlas_terms.json` | 唯一术语 ID、定义、别名、原理、来源 |

**禁止交叉复用项目数据。** `OpenSAE/SAEBase` 的任何 Tree 不引用 OpenResearch 的权威节点集合、旧图快照、演示研究任务或 其他项目专属的旧样例。OpenResearch 保留它自己的五棵 Tree；共用 CSS/JS 实现时必须使用项目标识、独立路径和独立持久化存储。公开 URL 为 `/opensae/saebase/atlas/{core,knowledge,code,environment,context}.html`，不能将 `/opensae/openresearch/atlas/` 重定向成 SAEBase。

## 2. 先 Context，后逐树映射

1. **锁定权威术语。** 收集两份权威术语 JSON 的全部术语与分类，合并时以稳定 ID 去重并验证定义、别名、`why_it_matters`、`foundation` 和 `source_evidence`。
2. **人工可理解的英文 Context。** 在 `doc/knowledge/i18n/context.en.json` 逐项维护英文 `title`、`summary`、`description`、`why_it_matters` 和别名；定义应准确呈现真实科学或工程含义，不得只写“参见源码”或“某组件用于某模块”。
3. **以 Context 引用建立关联。** Knowledge、Core、Environment、Code 的节点仅借助稳定 `context_term_ids` 指向 Context；不复制、随意重写同一术语定义。
4. **逐树补本地解释。** 先 Knowledge，再 Core，再 Environment，最后 Code。各树的专属 `title`、`summary`、`description` 以节点 ID 为键，在英语数据层完成，保留中文原始权威描述。
5. **验证后发布。** 全部节点语言覆盖、引用关系、页面资源及实际浏览器展示通过后，再部署网站并更新 README。

## 3. 节点数据与命名合同

- 每个节点有全项目唯一且稳定的 `id`；历史 ID 保留已有语义。需要改变语义时创建新 ID 和迁移映射，不凭展示标题推断身份。
- 每个节点有明确的 `kind/type`、所属 Tree、父边、自然语言 `title`、`summary`、`description`、独立 `status`，以及可以定位的 `evidence`。
- 父子边来自源权威图或实际 AST，Core/Knowledge 的研究关系与 Code 的 `contains` 不得混淆。一个可视化树节点应有确定的父节点，跨树联系使用明确引用。
- 标题是**有信息的研究/工程动作或概念**，应告诉人“做什么”或“定义什么”，不是函数名、文件路径、多个单词用间隔符拼接的清单。禁止 `foo · cfg · module · validate` 式报菜名。
- 中文和英文的标题均须语义完整、简洁自然；必要的 SAE、LLM、Gemma、Hydra、Token、API 等固有名词保持标准写法。英文标题必须准确表达函数职责，不以机械拆分 CamelCase、SnakeCase 当作最终说明。
- `summary` 应简洁说明对象的主要职责；`description` 补充输入、处理、输出、边界和来源，不能无条件复制 `summary` 作为所谓“完整解释”。可用权威来源不足时标记待审，禁止编造结果。
- 任何可见状态只能根据**该节点本身**的证据判定。源码接口已实现 ≠ SAE 科学假设已验证 ≠ GPU 测试通过。子节点状态不能自动把父节点染绿。
- 父节点展示时每页最多 **10** 个子节点，通过界面分页浏览所有真实实体。禁止按第 N 组、源码行号区间或数量临时生成 Tree 节点。任何新生成的功能节点必须拥有独立职责、可核验证据及有效的 Context 术语 ID；缺一则构建失败。
- 展示排序来自原始图边顺序或 Python AST 的实际位置。不得按 Work Plan 编号、临时 WP 名称、字母排序或 UI 随机值代替科学与代码结构。

## 4. Code Tree 特别规范

- 代码权威来源为真实 Python AST：模块路径、类名、成员属性、方法、参数、类型、默认值、返回表达式、函数主体、源码行号均来自实际源码。缺失不得猜测。
- `datasets/`、`models/`、`tasks/` 展开至具体实体，取消 领域分类等人为中间层；按源代码中的一等目录组织。
- 优先展示任务/模块在系统中的作用，再展示源码接口；标识符和完整签名放在节点展开详情，不能让它们主导节点标题。
- 每个可见的方法有参数与返回意义、必要的精确代码来源。代码、路径、标识符、哈希等原样保留，不进行机器翻译。
- 英文 Code 标题若通过生成器辅助创建，必须有明确的 AST 字段及语义规则，并通过“重名、空泛措辞、乱码、路径堆砌、作用误判”检查；实际科研关键节点优先使用已审定的特定英文说明。
- 不把历史兼容入口、文件移动、UI 导航组当作新算法。旧快照与当前代码矛盾时，以当前源码为准重建。

## 5. Environment 与实验真实性

- 环境数值仅从当前机器实测快照读取；静态公开示例必须保留 `synthetic: true`，并在中英文都显式提示“合成示例”。
- 无法观测的硬件、显存、版本或运行状态保留 `unknown`；不能因为没有 `nvidia-smi` 就断言无 GPU。
- 真实训练成功、生成有效、SAE 特征解释可信、性能提升必须使用独立运行收据支持；不能从代码节点存在推断实验结论。

## 6. 双语数据合同

- Tree 的 **中英文都必须完整**；英文版是强制必备，Markdown 的中文版允许不存在。
- 语言格式为 `project: SAEBase`、`schema`、`node_id`、`zh/en` 字段，使用 **Unicode UTF-8**。所有 `id`、源码、证据、算法、节点数和边不随语言变化。
- 英文权威数据按稳定 ID 编写到独立层，禁止前端基于中文字符串字典逐项替换充当主方案，禁止用“Source-linked detail for…”、“used by…”和同类空洞模板伪装科学翻译。
- `title`、`summary`、`description`、`why_it_matters`、别名、阅读说明、节点属性、函数参数含义和返回含义，均按可读语义覆盖；代码原文和确切来源信息豁免翻译。
- 找不到英文时**构建失败**并给出 `tree/node_id/field`，不能在正式页面退回中文、显示乱码或静默用模板填坑。中文可以缺省于非 Tree 文档，但 Tree 本身必须双语。
- 语言切换需要保留 Node ID、展开、Focus、Zoom、Review、搜索、URL 和文档锚点；所有五树共用一个语言偏好，并在各页面之间继承。
- 中英文字符串均通过 UTF-8 解码、U+FFFD、乱码模式与文本溢出检查。发布验收还要实际打开页面验证。

## 7. Markdown 与文档引用

- 从五树的证据、`foundation`、Glossary、代码接口、节点链接**自动枚举真实被引用的 Markdown**，形成 `source_path → English_path` 清单，避免手工漏项。
- 中文 Markdown 的英文版本使用同目录 `.en.md`（已有的英语文档可原样使用）；保留原有段落逻辑、引用、表格、公式、代码块、相对链接与 heading anchor。不能把中文原文或伪翻译说成完整英文。
- GitHub 链接一律指向**所属项目**的仓库和实际存在的版本；跳转到英文文档时保持同一节点与锚点。未经明确指令不泄露或外链 private repo。
- 历史工作计划与 GitHub 总结文档保留在 `harness/github_work_plan/`、`harness/github_development_doc/`，普通任务 Agent 不默认全量读取；用户可见的文档解释由 Tree 的权威关联精确选择。

## 8. 生成器、快照与发布流程

1. 校验上下文源文件、AST、引用路径、Graph 节点 ID、Parent、Status、Context 的关联和每个英文术语。
2. 生成 Code Tree 与五树数据，保证输入源码版本明确且可复现；生成物包含 provenance 和源提交身份。
3. 依赖生成的真实节点逐项生成/审定英语内容，并检测覆盖率。**不得通过 UI 现场翻译弥补构建缺失。**
4. 运行 Source/Locale/Browser tests：节点数、层级 fanout、两个语言切换、断链、Unicode、重要方法与 Context 定义完整度、点击交互、翻页和状态保持。
5. 只将 SAEBase 数据发布到 `/opensae/saebase/atlas/`，OpenResearch 继续使用 `/opensae/openresearch/atlas/`；部署 HTML、JS、CSS、所有数据分块和 manifest 在同一版本中完成，校验每块内容哈希，避免半更新。
6. 对 SAEBase 网站和 OpenResearch 网站分别访问五个 `*.html?lang=en` 与 `?lang=zh`，确认页面标题、实际 graph.project、所有资源加载和内容来源。README 的五条链接必须指向 SAEBase 专属网址。
7. 源分支、发布分支和工作计划使用 PR 审查；独立 CI 和站点验收绿色后才合并 `main`。保留前一可工作的快照及回滚入口。

## 9. 失败条件与验收清单

**必须失败**：错误项目 ID、混入 其他项目专属的旧研发内容、跨仓库错误链接、丢失 Context ID、英文缺字段、含中文的英文说明（技术引用除外）、乱码 U+FFFD、杜撰的源码或实验结论、导航断边、丢失发布 JS 文件、过大的单层子节点、站点分块哈希不匹配、浏览器节点切换失效。

**可接受**：代码片段原文包含中文注释但明确标注“源码原文”；文档只存在英文；Environment 资料标记 unknown；尚未开展 GPU 科学实验而状态保持未验证。

验收报告记录两个仓库的精确 commit、各 Tree 节点数、双语覆盖、文档映射数量、CI/浏览器验证方式、两个独立公开 URL 的检查结果和未通过事项。规范文件自身与构建器须相互引用，防止未来 Agent 重复把两个项目混在一起。

## Code Tree 功能概念分层合同

- `saebase/` 内部先按研究与工程职责进入 14 个已审定概念领域，再按有明确输入、处理或输出边界的功能细分。概念目录维护于 `harness/code/atlas_concepts.json`，实现于 `harness/code/functional_taxonomy.py`。
- 概念节点属于**有源码证据的功能层级**，不是任意导航占位。每个概念必须有独立作用、权威 Context ID、覆盖真实 Python 文件的 evidence、准确的中英文标题与摘要；分类缺失、重叠、空概念或语义重复均应拒绝。
- 每个节点真实的直接子节点**不得超过 20**，优先保持在 **10** 以内。显示分页只解决屏幕空间，不能代替功能分层。禁止按文件数量、源码行区间、字母顺序、Work Plan 编号或“其他接口”创建概念节点。
- 功能概念允许跨包组织多个实际模块，但原 AST 节点 ID、签名、定义位置、子方法及引用边保持稳定。分类过程必须验证输入模块与输出叶节点一一对应，新增真实源码应先得到符合 Context 的归类，再通过生成验收。

- 单纯转发旧 Python 导入路径的兼容模块不构成独立科研功能，Atlas 隐藏这些重复展示，但不得删除或破坏源码的兼容行为。真实数据处理实现仍由 Dataset Tree/原始 Python 模块提供。
