# 运行环境资料

## 目的和主要功能

说明本机解释器、依赖和资源资料的来源，帮助复用实验环境和定位运行条件问题。

## 适用范围

用于了解实验记录中的实际环境，或按用户要求采集、比较环境资料。

当前科研运行的解释器与依赖选择由 `conf/runtime/environments.yaml` 和 `saebase/runtime/environment.py` 维护。运行记录保存实际解释器、依赖版本及来源，读取时结合记录时间和进程范围使用。

中央环境采集与比较由 ResearchHarness 提供。通过 `.researchharness/central.json` 的 `harness_root` 找到 `src/harness/environment/environment.md`；该说明介绍环境快照的采集、保存、选择和比较，适用于明确要求维护环境资料的任务。按该入口和授权选定快照，保留采集时间、来源及合成示例身份。

运行条件缺失时定位具体依赖、设备或解释器及其影响，在已授权范围内处理；后续任务复用负责模块已经确定的环境事实。
