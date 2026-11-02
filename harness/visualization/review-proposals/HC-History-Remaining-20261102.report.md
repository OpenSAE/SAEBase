# 历史资料清理：补充人工审批

项目：`OpenSAE/SAEBase` ；基线：`e9659908992608a98244813a11363cfd90cbbc9f`。

已合并的 [PR #162](https://github.com/OpenSAE/SAEBase/pull/162) 完成 55 个原路径真正迁移（62 条档案记录），尚有历史文档与现役入口相互引用。之前 [Human Check #161](https://github.com/OpenSAE/SAEBase/issues/161) 没有覆盖以下路径；此处将它们独立列为只审批文件的 G01–G04。**不改变 Atlas 的节点标题、说明、性质、状态、关系、动画、颜色和审阅交互。**

## 清理目前仍引用旧资料的项目说明

删除或替代现役说明中的旧实施步骤、历史结果作为当前权威的链接，保留当前真实功能、示例和来源。

- `README.md`
- `README_cn.md`
- `saebase/research_lab/README.md`
- `saebase/generate/README.md`
- `tasks/generate/backend/README.md`
- `saebase/research_lab/frontend/index.html`

验收：只更新文字和历史链接；不得更改 Research Lab 页面交互及排版；现役功能描述保持真实；版本证据通过固定 Release/Tag 查阅。

## 解除发布工具对旧文件的要求

去掉仍从历史文档或归档工作表读取执行规则的代码路径，用可执行当前合同或只读固定版本证据替代，防止软件发布和当前测试因归档失效。

- `tasks/benchmark/release/gates.py`
- `.github/workflows/m5-release.yml`
- `saebase/tools/wp12_p8_acceptance.py`
- `saebase/tools/stage_d_archive.py`

验收：不放宽当前发布、数据安全及校验规则；既有实验测量、评分、解析语义不变；已有功能测试仍通过；退役行为按 Release Tag 隔离。

## 更新当前考试和执行规范的入口

现役 Agent 和考试规范不应强制读取冻结的 WP18–21 Markdown；保持历次考试数据不变。

- `harness/contracts/README.md`
- `harness/contracts/wp20_design_validate.py`
- `harness/tests/wp18/README.md`
- `harness/tests/wp19/README.md`
- `harness/tests/wp20/README.md`
- `harness/tests/wp21/README.md`
- `harness/tests/README.md`
- `.cursor/rules/core/10-context-selection.mdc`

验收：不修改任何冻结考试成绩、真实测试收据与标准答案；维持原有 Agent Harness 审批及安全规则；历史考试仅作为冻结版本复现，不作为新任务依赖。

## 防止旧依赖重新进入当前代码

为非归档现役代码和持久化说明补上不依赖 Archive 的自动检查；历史文件独立接受完整性检查。

- `saebase/tools/check_documentation.py`
- `harness/github/work_test/test_repository_architecture.py`
- `.github/workflows/ci.yml`

验收：不影响既有 CI 选择规则和正常性能；静态扫描豁免只读历史收据、Release 固定版本来源；干净构建在删除 archive 目录副本后也能运行。


## 严格边界

1. 当前文件必须能独立运行：旧文档不能再作为当前代码、普通 CI 或持久化说明的依赖。
2. 冻结考试运行记录（`harness/tests/wp19/runs`、`wp20/runs` 等）以及公开 Release 历史保持不变；不可伪造旧成绩或修改已经完成的实验。
3. 本次人工审批只授权列明的现役文件范围。其他文件若要改，应另行审阅。
4. 公开 Atlas 为浏览器草稿，必须点击 Send report 并由用户亲自提交 GitHub Issue；不得推断自动认证批准。
5. 实施与最终合并以真实 CI、发行构建及独立测试结果为准。
