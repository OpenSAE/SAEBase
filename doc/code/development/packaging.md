# 安装与源码布局

## 目的和主要功能

说明当前 SAEBase 分发包与源码的对应关系，以及仓库外安装的必要检查。

## 适用范围

适用于构建、安装、维护 Python 包及配置资源；科研任务按自身资源条件执行。

## 包含内容

根级 models、tasks 和 conf 按现有结构安装。数据实现保持在根级 datasets，构建时放入 saebase.datasets 命名空间，源码使用同一份物理实现。第三方 Hugging Face datasets 保持独立。

安装包包含当前科研实现、配置和必要资源，不包含权重、本机运行目录或历史测量归档。Research Lab 的 Web 依赖通过 lab 可选安装提供。科研核心不依赖 ResearchHarness。

## 当前命令

提供 saebase、saebase-prepare-prompts、saebase-check-env、saebase-intervene、saebase-benchmark、saebase-audit、saebase-ground-truth 七个命令。多模型与 SAE 比较通过评测命令读取比较配置。软件发布由当前独立发布流程处理。

## 检查范围

在源码目录外安装生成的 wheel，读取模型、数据、任务登记与配置资源，并执行明确选定的离线例子。sdist 需要能够重新构建同一实现。检查和构建按用户指定范围运行；源码存在不等于安装或科学验收已经通过。

组件预设使用 `conf/<组件>/catalog.yaml`，由 Hydra 的具名配置登记保持选择语法；直接读取使用 `conf.load_group`。构建前清理所属包的旧构建输出，使删除或迁移的源码准确反映在新包中。分发检查确认实际资源和安装调用，配置数量按当前组织方式统计。
