# Codex 客户端配置

## 目的和主要功能

保存 SAEBase 的客户端权限、常规模式中央接入和按需诊断入口。

## 适用范围

用于客户端接入、模式切换和权限故障。任务方法由根及局部 AGENTS.md 和相关技能说明。

- `config.toml` 配置客户端权限，`rules/default.rules` 保存命令规则。
- `hooks.json` 注册本地文件和命令的权限入口 `hooks/tool_policy.py`。入口读取当前模式：密集开发使用客户端权限和项目路径范围，常规模式直接调用中央公共接口。
- `hooks/development_mode.py` 使用标准库读取项目模式，使密集开发独立于中央安装。
- `hooks/preflight.py` 在接入、配置变化或权限故障时按需读取客户端实际配置，使用 `python .codex/hooks/preflight.py` 调用。

语法检查在相关修改完成后按文件范围执行；发布持续集成使用 Python 自带的 `compileall` 检查科研源码，已有测试检查实际功能。
