> **树的生成、写作、翻译及发布必须首先遵守**：[SAEBase Atlas 生成与编写规范](ATLAS_AUTHORING_STANDARD.md)。OpenResearch 使用独立来源和网址。

# SAEBase Atlas · 节点内展开式知识浏览

Core、Knowledge、Code、Environment、Context 共用一套居中分叉、Focus Mode、颜色与节点属性渲染。Core、Knowledge、Code、Context 聚焦有来源的现有功能与方法；Environment 显示现场观测或明确标记的演示配置。

## 交互

在 Overview 中点击节点右下角的子节点数量徽标可展开或收起其分支；点击任何卡片主体会在原树内展开对应子节点和节点自己的说明，保留已经展开的其他分支。点击节点底部的 **展开节点属性与依据**，卡片在当前树中变宽、变高，邻近节点按照子树高度重新居中排列。展开内容位于同一张卡片内，类中的各个方法还可逐项打开。节点采用稳定的显示比例，树变大时通过画布滚动查看；Fit 是用户主动缩放操作。

- **函数**：主要作用、签名、每个参数的名称与含义、类型、默认值、返回表达式与含义、真实源码主要部分。
- **类**：职责、全部成员属性、类中定义的每个方法。每个方法拥有完整的函数接口和源码。
- **模块**：包含的类、函数，以及当前模块中主要的源码部分。
- **Context**：简洁定义、原理、来源、现有实现；适合的概念附带公式、SVG 示意图和真实代码。
- **Knowledge**：LLM SAE 的八个研究方向；区分尚待验证的研究问题与已经有代码证据的方法接口。

界面文字采用肯定式，优先显示直接职责与证据。实际源码按文件和行号链接，术语及模型说明由 [Context 的权威数据](../../doc/knowledge/) 维护。

## 本地生成

```bash
python harness/core/validate.py
python harness/knowledge/validate.py
python harness/knowledge/context_attributes.py
python harness/code/build.py
python harness/visualization/export.py
python harness/visualization/build_pages.py
python -m http.server 8765
```

从 [Code Tree](https://atheavenma.github.io/opensae/saebase/atlas/code.html) 或 [Context Tree](https://atheavenma.github.io/opensae/saebase/atlas/context.html) 进入。公开 Environment 采用合成示例设备。

公开 Atlas 存放在独立的学术主页仓库。构建器会生成完整的 `data.js` 和较小的 `atlas-structure.js`。后者以权威 Knowledge 和 Code 分组结果更新已发布的大型快照。发布 Atlas 更改时同步 `app.js`、`style.css`、`atlas-structure.js` 及 HTML 资源版本；如需完整更新代码实体和源码说明，再同步 `data.js`。

## 组件边界

Code Tree 按真实源目录 → 子模块职责 → Python 模块 → 类／函数 → 方法排列。每页展示最多 10 个真实子节点，超过时由界面分页，不再创建第 N 组或虚构的导航实体。新增功能节点必须具备独立的源码职责、真实证据和权威 Context 关联；节点数量不构成新增功能。Code Tree 将通用读取接口、模型构建及任务调度放在各自的公共层，将数据集特定的读取与预处理放在「具体数据集 → 对应数据集」下，模型结构和兼容性放在「具体模型 → 对应模型」下。七类 SAE 操作分别在「具体任务 → 对应操作」下展示，任务配置独立展开；模型与数据集通过运行配置组合，不写入通用 Task 的名称或定义。

## 节点标题的语义约定

Atlas Code Tree 的节点名称必须是一句有内容的中文功能概括，例如「校验 Hydra 训练配置与注册组件」「计算模型激活的重建损失」「为输入样本附加 Token 编码」。科研约定名称 SAE、Hydra、Token 等可按惯用缩写书写。

标题只回答「这个节点做什么」。Python 的函数名、类名、方法签名、文件位置和源码行号存放在**展开节点说明与依据**中，使用原始源码身份保证可复查。Code 节点从实际 Python AST 提取，`harness/code/semantic_titles.py` 负责按功能生成中文标题；`harness/code/presentation.py` 负责保留经过核对的中文说明与源码接口。

各研究操作优先连接到 Context 中对应的术语定义。Code Tree 的展示分层仍然保持每个父节点最多十个直接子节点，避免以源码路径的字符串拼接替代语义分类。

## 节点解释与术语来源

每张卡片点击后会在原节点内部展开「这个节点在做什么」，说明节点的职责与可核对的来源。出现的专业词汇会在「专业词汇 · Context」直接显示简短定义，并链接到 Context 中的权威词条。

科学术语统一来自 `doc/knowledge/glossary.json`，源码、任务与执行环境术语统一来自 `doc/knowledge/atlas_terms.json`；两套词条经 Atlas 导出器共同发布到 Context Tree。其他视图只保留稳定 `context_term_ids` 引用，绝不另写第二份定义。新增术语须同时填写定义、别名与仓库内证据路径，并通过自动检查。

## Atlas 内审阅

审阅不另开页面。**待审节点使用与普通节点完全相同的卡片、展开属性、方法签名、参数/返回值、公式、源码、Context 关系与 Details。** 只有状态底色不同：紫色 Unchecked、红色 Approve、灰色 Deny；红或灰可随时点击 Unchecked 恢复紫色。

底部左侧是 All → Tree → Parent → Node 三级分类筛选，按源码树的 DFS 顺序排列。每个分类包含分段环形进度及红/灰/紫各自数量。四个箭头中，最左与最右是 Previous/Next Unchecked，内侧是强制 Previous/Next。快捷键为 Enter / Backspace / U，方向键 ← → 搜索未审阅，Shift+← → 强制相邻翻页；输入框焦点时不拦截快捷键。背景以轻量 CSS 网格缓慢移动，尊重 reduced-motion。

- [四树演示](knowledge.html?review=OR-Atlas-MultiTree-Demo-20261026)：Knowledge → Code → Environment → Context，同一页面滑动切换，可查看方法签名、GPU facts 和术语别名。四项全都是测试提案，不自动改变代码或设备数据。
- 本地实时写入：`python -m harness.core.agent_flow serve OR-Atlas-MultiTree-Demo-20261026 --port 8766`，使用终端显示的完整 URL
- 每次点击同步保存 Plan JSON，空 Knowledge 新节点批准时同步写入 Graph/Glossary，退出页面不丢失已保存的决定
- 公开 GitHub Pages 页面中的 Approve / Deny 会实时修改颜色、在浏览器的 JSON 草稿存储中保存决定并自动翻页；底部 JSON ↓ 可导出文件。这些决定并未写入 GitHub 权威图。
- 本地认证的 Atlas 服务可在每次点击后同步更新审阅 Ledger、Knowledge Graph 与 Glossary。
- 每个节点的 partial 状态独立审计，不自动沿父分支传播；仅作背景的上级节点在审阅界面保持中性配色。

[执行治理合同](../core/AGENT_FLOW.md)

## 科研概念驱动的 Code Tree

`saebase/` 不再将数百个 Python 模块直接罗列在同一父节点下，而是在 Code Tree 中建立 14 个有证据和 Context 关联的科研概念，再细分为实际操作。`harness/code/atlas_concepts.json` 逐个定义分类的中英文作用、源码匹配规则及术语关联。每个概念严格保留真实源码 ID、签名和路径；超出 20 个直接子节点须按功能进一步细分，优先控制在 10 个以内。页面的每页 10 个节点仅用于交互分页，不代替科研层级。
