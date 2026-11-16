# 验证与科研复现

## 目的和主要功能

说明按具体改动选择检查、解释证据，以及在相同科研条件下复用和比较结果的方法。

## 适用范围

用于功能检查、科研复现和软件发布准备。各任务的输入和完成要求由其正式接口及协议定义。

## 根据用途选择检查

开发时使用能回答本次问题的功能场景。先确定结果要支持的动作，再从实际入口生成、读取并使用它；需要保存的内容在原执行结束后重新读取。相关未完成场景包括实际对象、已知原因、影响和下一步，调用者据此修复条件、继续任务或使用已有成果。

完整离线测试、安装与打包检查由[发布验证](../../../.github/workflows/release-validation.yml)和明确需要整体检查的任务采用。模型及 GPU 实验按对应任务选择资源与预算。

## 理解已有证据

登记资料说明可选择的配置，合成输入说明所检查的软件行为，真实模型记录说明对应权重、数据与协议的观测。科研结论使用符合其问题和协议的证据。历史记录按原日期和输入阅读；当前模型资料见[模型与 SAE 支持](model-support.md)及[逐层观测记录](verified-model-matrix.md)。

## 复用与比较

采用相同模型与 SAE 身份、输入、位置、划分和指标定义；关联样本按原分组计算不确定性。缓存及恢复使用所属模块已有的身份检查。条件不同的结果保留差异和不可比较的原因，所用来源、配置与原始数值保持可追溯。

检查资料说明实际对象、观察及后续用途。软件发布见[发布流程](release-process.md)。


以下保留原日期的科研观测摘要，内容由现有模型支持资料生成器维护。

<!-- BEGIN VERIFIED_MODEL_SUPPORT -->
## Verified model and SAE evaluations

**18** selected model-SAE configurations, **48/48** real-checkpoint compatibility tests, **48/48** benchmark evaluations on 64 held-out prompt groups, and **48/48** reliability audits (2026-10-16). See the [per-layer reports](verified-model-matrix.md).
<!-- END VERIFIED_MODEL_SUPPORT -->
