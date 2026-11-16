# 项目工作模式

## 目的和主要功能

说明如何在持续密集开发和常规中央管理之间切换，以及如何查看实际生效的设置。

## 适用范围

适用于 SAEBase 的智能助手开发任务与明确的模式切换。科研算法、数据格式和 Task 接口继续使用现有实现。

## 密集开发

用户说明开发目标后，助手直接修改、重构、运行必要检查并修复问题，完成后报告结果。修改前人工审阅默认关闭，中央计划、Flow 和临时维护审批暂停。模式持续生效，用户主动切回时结束。

```bash
rh --harness-root /workspace/ResearchHarness --project saebase mode set intensive
rh --harness-root /workspace/ResearchHarness --project saebase mode status
```

客户端开发设置使用 `default_permissions = ":danger-full-access"` 和 `approval_policy = "never"`。它使用当前账号的实际权限。项目 Hook 读取本地模式后直接执行开发检查；密集模式读取无需安装中央软件。用户的实际任务决定开发内容，科研身份、样本划分和实验结果如实记录。

通过中央 `security run-project` 启动时，密集模式保留调用管理员已有的开发身份；常规模式沿用研究账号隔离。直接启动的进程继续使用其原有账号。

## 恢复常规管理

```bash
rh --harness-root /workspace/ResearchHarness --project saebase mode set regular
```

切换程序保存原有管理设置，只恢复客户端中与权限选择及审批有关的字段。科研代码、新增配置、模型选择和实验结果保持当前内容。已有命名权限配置统一由 `default_permissions` 选中，旧式 `sandbox_mode` 选择合并为一致配置。

模式保存在本机忽略的 `.researchharness/mode.json`；恢复资料在中央 `.workspace/harness/core/development_mode/saebase/`。重复选择同一模式保持幂等。设置写入失败时恢复本次切换前的配置。

## 检查实际客户端

Hook 每次读取当前模式；新的客户端会话读取更新后的配置。已运行会话需要按实际情况重新加载。可运行 `python .codex/hooks/preflight.py` 查看客户端实际读取的模式、审批与权限配置。SessionStart 在密集模式下直接说明工作方式。

客户端配置依据 [OpenAI Docs 配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)：`default_permissions` 与旧式沙箱选择分别使用，`approval_policy=never` 用于非交互运行。
