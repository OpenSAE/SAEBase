# Precheck 资源准备与实际检查

## 目的和主要功能

准备并检查选定数据、模型、SAE 与任务条件，提供执行判断和具体原因。

## 适用范围

用于明确选择的 Precheck 或正式任务的自动检查；资源动作按当前请求与配置处理。

Precheck 使用 SAEBase 唯一总控 `conf/config.yaml`，复用现有 Dataset、Model、SAE 和科研操作接口。没有新增下载任务、管理任务或 Registry。

## 执行流程

`task.execute=false` 或 `--cfg job` 保持配置用途，不进行安装、下载或计算。设置 `task.execute=true` 后：

1. 解析准确的模型、SAE、层位和检查目标。
2. 尝试安装缺少的分析依赖，固定已安装的 Torch、NumPy 等模型数值栈版本，并在新进程验证导入。
3. 优先检查本地完整缓存；缺少文件时尝试下载选定模型及所需 SAE 文件，记录仓库版本和缓存位置。一个 SAE 检查不会下载同仓库全部层位。
4. 未给出显式提示时，通过选定 Dataset 的准备接口补齐数据，复用已有生成、编码、过滤和 manifest 规则。
5. 检查数据产物，再运行选定的 SAE 或任务检查。资源准备完成不等于实际检查通过。
6. 保存准备记录、原始日志、检查结果和产物审计。

```bash
saebase task=precheck task.execute=true \
  task.mode=task 'task.check_tasks=[capture]' \
  task.device=cpu experiment.sub_name=basic
```

Dataset 和 Model 仍从总控的 `dataset=...`、`model=...` 选择。`task.mode` 支持 `sae`、`task`、`all`。任务检查列表为空时沿用全部七个科研任务；需要原生计划的 benchmark、circuits、explain、train 必须提供实际参数，Precheck 不生成虚构计划。

## 默认准备策略

- `auto_setup`、`allow_network`、`allow_install`、`allow_download` 默认开启。显式关闭网络或准备功能仍可用于离线诊断。
- 模型与 SAE 的缺失文件下载预算默认 16 GiB；准备超时 1800 秒，数据准备另有 1800 秒。实际检查仍使用 `max_seconds`、`max_samples`、`max_tokens`。
- 默认数据准备生成 32 个候选、要求至少保留 1 个。它是明确记录的最小检查数据准备预算，不改变 Dataset 原有过滤条件，不代表完整评测数据或科研验收。
- 已有完整数据直接验证并复用；数据准备采用锁与临时目录提交，已有不完整产物不会被覆盖。
- 设备由 `task.device` 指定，实际检查计划由 `task.task_configs` 指定。删除 `allow_compute`、`allow_gpu`、`allow_native` 二次确认开关；执行请求不会在下载后再等待人工确认。
- 原生训练检查保留最多两步等实际预算要求。缺少输入计划或硬件不可用会报告具体原因。

## 记录与失败

结果保存在当前 Runtime 运行目录：

- `metadata/precheck_result.json`：准备阶段及实际检查结果。
- `metadata/precheck_audit.json`：报告文件完整性检查。
- `tmp/precheck_jobs/<阶段>/metadata/`：请求和子进程回执。
- `tmp/precheck_jobs/<阶段>/logs/`：完整标准输出、错误输出，以及安装、导入和下载事件。

读取时先查看请求范围、阶段及实际检查对象，再读取对应结果与日志。结果中的具体原因和处理建议用于补足依赖、资源或输入，已经完成的内容按所属任务协议复用。开发检查与记录功能时，同时安排这些内容在运行结束后的读取和实际处理。

安装、来源访问、下载、数据准备、预算不足等准备失败分别记录 BLOCKED 原因。软件执行错误仍保留 FAIL，不伪装成环境问题。超时停止整个工作进程组，避免安装或原生计算子进程留在后台。

这些记录可以证明软件执行到哪一步；不能自动证明模型性能或独立科研验收。离线测试使用明确标记的合成响应和模型，不替代真实链路运行。
