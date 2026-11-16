# SAEBase 贡献与开发

## 目的和主要功能

说明代码、文档和科研接口的维护方式，以及面向 GitHub 的贡献与发布入口。

## 适用范围

适用于 SAEBase 开发和外部贡献。智能助手按根 AGENTS.md 的当前工作模式及用户任务执行；公开贡献使用下述 GitHub 流程。

## 完成一项修改

围绕一个具体问题修改现有负责模块。制定功能方案时，说明调用者或后续研究将怎样使用结果，据此安排输入、检查、返回信息及需保存的资料；保存与读取同时设计，相关失败情形提供具体原因、影响和后续处理方法。保持模型、SAE、数据划分和结果来源清楚。目录职责和测试原则统一采用根 AGENTS.md。

选用能确认本次功能结果的现有检查，需要补充时使用最小输入。辅助代码按持续用途纳入正式实现或长期测试，完成临时用途的代码随任务清理。

开发环境按需安装 `python -m pip install -e ".[dev,quality]"`。局部开发运行相关测试，项目级验证由持续集成、发布准备或具体任务安排。文档修改检查实际变化的内容与链接，已有文档检查入口是 `python scripts/documentation.py --check`。

## 公开贡献

问题单用于报告问题和讨论需求，拉取请求用于提交可供维护者合并的代码。按贡献需要提供问题背景、实现效果、实际检查结果和涉及的模型或数据来源。

新公共接口使用 `saebase.api` 与 `saebase.cli`。接口或持久化格式变化时，同步迁移实际调用者并说明影响。安装与资源布局见[打包说明](../doc/code/development/packaging.md)。

## 科研与软件结果

合成输入用于检查软件行为；真实权重、数据和科研协议的结论以实际运行结果为依据。第三方源码保留其来源与许可。

## 发布与质量工具

软件发布按明确的发布任务执行，参见[软件发布流程](../doc/code/development/release-process.md)。版本、发布说明和实际发布产物保持一致。

[代码质量说明](../doc/code/development/code-quality.md)说明局部格式与类型检查；[持续集成说明](../doc/code/development/ci-maintenance.md)说明日常检查、中央接入与发布验证的用途。

## 文档与代码审阅

功能说明直接解释处理对象、输入、计算、输出和使用方法，必要术语结合实际对象解释。现行操作在对应模块集中维护，其他文档按阅读需要连接它；README 介绍能力，任务说明保存协议，执行记录保存一次维护的过程。

拉取请求围绕具体问题及实际使用审阅。检查真实调用、保存后读取、未完成原因与后续处理，接口变化同步处理调用者。CODEOWNERS 指定路径的审阅责任，仓库保护设置决定合并要求；维护者读取当前分支与适用持续集成结果，再按本次贡献处理。完整发布采用软件发布流程。


## 科研与工程操作的归属

数据准备使用 `saebase-prepare-prompts`，环境查询使用 `saebase-check-env`。科研流程和检查分别进入对应 Task；数值参考使用 `python -m saebase.api.reference`，真实配对检查使用 `python -m tasks.precheck.sae_checks`。评测短协议通过 `saebase-benchmark --pair-protocol cpu_pair --sae-key <名称> --real --device cpu` 或 `--pair-protocol reference_pair --model <名称> --real` 选择，参数保存于 `conf/benchmark/catalog.yaml`。这些短协议保存自己的输入数量、预算与统计方式。

工程维护使用以下四个入口：

- `python scripts/quality.py`：选定代码的格式和静态检查；`select --base <提交> --head <提交>` 选择持续集成检查。
- `python scripts/documentation.py --check`：文档链接；`support --check` 或 `support --write` 读取评测证据并检查或更新支持表格。
- `python scripts/distribution.py contents|install|roundtrip <目录>`：分别检查分发内容、仓库外安装和源码包重建。
- `python scripts/release.py --help`：版本来源、发布条件和资产登记。

已保存的评测证据使用 `python -m tasks.benchmark.results.archive --help`；具体运行的指标和身份核对使用 `python -m tasks.benchmark.results.evidence --output <运行目录>`。来源文件查询使用 `python -m tasks.benchmark.sites.catalog sources --output <文件>`。中央查询直接使用明确绑定的 `rh` 命令；中央引擎身份的指定验收归集成测试。

Lab 的解释、私有执行、公开包准备和电路展示接入由各自后端提供，依次使用 `python -m saebase.research_lab.backend.auto_interp_local`、`private_worker`、`publish bundle` 和 `published_records` 对应模块的帮助。
