# Agent 执行任务的流程 · Atlas 人类审阅

所有审阅直接在现有的 Atlas Tree 中操作。Core 与 Harness 规则不属于普通任务 Agent 的可修改范围。

## 审阅新的空节点

- **紫色：Unchecked。** 外部 `github_work_plan` 的提案还没有人类决定。
- **Approve → 红色：Not Implemented。** 审核通过了节点的存在与位置，不代表代码实现，也没有科研结果。
- **Deny → 灰色：Denied。** 在审阅 JSON 中保留否决记录，节点不进入正式 Knowledge 图。
- **Unchecked → 紫色**：可以把已经 Approve 的红色或 Deny 的灰色节点恢复为紫色，撤销浏览器草稿中的决定。本地服务对空 Knowledge 节点也会安全撤回其已批准的占位写入。
- 在现有 Atlas 的 All / Tree / Parent / Node 三级筛选中按源树 DFS 导航；每级显示红/灰/紫计数及环形进度。最外侧箭头寻找未审节点，内侧箭头强制翻页。快捷键 Enter / Backspace / U / ← / → / Shift+← / Shift+→。
- 点击状态决定后先验证、写入 JSON 再自动跳到下一个待审节点；Unchecked 重置则停留在当前节点，方便确认恢复成功。

在仓库根目录运行：

```bash
python -m harness.core.agent_flow inspect OR-Knowledge-Demo-20261026
python -m harness.core.agent_flow serve OR-Knowledge-Demo-20261026 --port 8766
```

浏览器打开终端打印的完整地址，其中 `#key=...` 只在当前本地审阅服务会话有效。**不要将包含 Key 的地址提交到 GitHub。**

每次通过/否决都会立刻写入 `harness/github_work_plan/agent_flow/<ID>.json`；通过合法空 Knowledge 新节点，还会写入 `doc/knowledge/context_graph.json` 和 `doc/knowledge/glossary.json`，执行原有 Knowledge/Glossary 验证。服务器返回保存成功后才会自动切换节点。关闭页面不回滚已确认的操作。每次点击都受 JSON revision 和来源 SHA 限制；若其他程序改过权威图，将拒绝过期的审阅。

## 与科研实验的审批是两件事

同意加入空节点不产生代码、实验或任何绿色结果。后续正式科研执行仍按独立的 Agent Flow 协议：规划审计、精确文件和命令范围授权、实际代码审阅、实验和独立科学结果审计，再由人类逐节点批准完成程度。黄色对应已部分实现且待测试的能力；绿色必须有真实结果和人类核验。

Core、`.cursor` 及 Harness 治理规则始终屏蔽于普通 Agent 编辑范围，非当前授权任务下的代码和 Knowledge/Context 修改不得自行执行。

## 本地和 GitHub

本地审阅使用 loopback Atlas 服务器实现即时写入。服务仅绑定 `127.0.0.1`，POST 检查 Origin、一次会话密钥与 revision，JSON 操作串行执行。它不是 OS 级安全沙箱；机上仍需要可信的操作用户权限。

GitHub Pages 是静态 HTML，没有安全的仓库写入身份。公开 Atlas 可以正常执行紫色→红色/灰色的本机草稿操作，每次点击把审阅 JSON 存到当前浏览器并自动切换节点；JSON ↓ 导出该草稿，但不直接写入 GitHub 权威仓库。本地审阅后的变更可以通过经过代码评审的 GitHub PR 发布；GitHub 原生 `.review-decisions.json` 回执入口仍可用于正式任务流程，但不能把其审批误作空节点的科研完成证明。

## 测试

```bash
python -m unittest discover -s tests -p "test_atlas_review.py" -v
python -m unittest discover -s tests -p "test_agent_flow.py" -v
node tests/test_atlas_interactivity.cjs
python harness/knowledge/validate.py
```

只有本地实际的点击、刷新和来源图校验完成后，才能将这一交互功能标记为通过完整人工验收。

## 状态独立性与卡片尺寸

每个 Knowledge 节点只根据其自身的证据维护状态。子节点变成 `partial` 不会自动改变父节点或更上层问题的状态。在 Atlas 审阅期间，仅作为背景的父节点使用中性样式；紫色 Unchecked、红色 Not Implemented、灰色 Denied 只标记正在审阅的节点。卡片尺寸与全部展开属性都使用正常 Atlas 节点的渲染逻辑，预审状态只影响颜色，不替换任何详情功能。

对于未配置写入后端的公开网页，颜色与本地 JSON 草稿是可持续的**审阅草稿**，不是源数据事实；要更新项目 JSON，使用本地服务或把导出的草稿交由受授权的人类审核合并。

## 四棵 Tree 的实例审阅

使用 `OR-Atlas-MultiTree-Demo-20261026` 可以检查 Knowledge、Code、Environment、Context 之间的连续翻页（Core 不参与）。同一画布切换视图和节点，保留 Atlas 聚焦、缩放与进度。

- Knowledge：`g.cub.agent_review_test`，位于分类训练方案下，具备科研问题、输入、预测输出和验收占位。
- Code：在真实 SAEBase 类与方法节点下审阅新接口提案，标注参数、返回值和实际源码依据。
- Environment：`env:atlas_test_gpu_profile`，位于 GPU 分支下，设备型号、显存和驱动信息均为显式测试值。
- Context：`term.atlas_test_calibration`，位于 Training 术语分组下，提供示意定义、公式、别名与用途。

公开页面每个 Approve/Deny 只保存浏览器 JSON 草稿；本地审阅服务保存 Plan Ledger，批准知识节点时同步更新 Knowledge/Glossary。其他测试提案不写入实际模型源码、设备探测或术语权威源。严格禁止将示意字段作为真实测量证据。
