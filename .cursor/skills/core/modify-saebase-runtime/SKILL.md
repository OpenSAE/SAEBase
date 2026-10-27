---
name: modify-saebase-runtime
description: 当用户明确授权修改 saebase/runtime 的公共注册、配置、训练基础设施、summary 接口或底层语义时使用。
---

# 修改共享 Runtime 的受控流程

1. 先判断需求是否真正对多个 Dataset、Model 和 Task 通用；局部需求优先放在业务模块。
2. 映射受影响的接口、直接调用者、YAML 参数及可能改变的默认行为；给出向后兼容方案与回归检查。
3. 未获用户明确批准，不编辑 saebase/runtime。路径 Hook 默认阻止 Agent 编辑 saebase/runtime；如确需修改，由人类在 Cursor 进程环境中开启 OPENRESEARCH_ALLOW_RUNTIME_EDIT=1，再重启/运行本次受控编辑。不得自行绕过 Hook。
4. 尽量保持函数签名、运行语义、seed、日志格式与 checkpoint 兼容；无法兼容时明确写出迁移方案。
5. 验证 python .cursor/hooks/validate_repo.py --full，并运行受影响的 Dataset/Model/Task 合同测试和必要的最小训练。
6. 汇报受影响范围、兼容风险、实测结果以及哪些运行时场景仍未验证。
