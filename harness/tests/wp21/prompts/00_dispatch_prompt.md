# 总调度提示

WP21.1 调度前必读 `harness/tests/wp21/EXAM_READY.md`。新工具包含 `suite_lock.json`、`probe.py`、带 `--control` 的 `record.py`、独立 `review.py` 及 `summary.py`；真实资源不足保留 `ENV_BLOCKED`，所有真实题目仍为 `NOT_RUN` 直至执行。

你负责安排本包 WP21 的本地真实 SAEBase 综合测试。只有用户明确要求开始执行时才分派；如果用户只要求阅读/准备，保持 prepared_not_assigned。

使用本包完整源码与冻结 cases.json。先读 harness/tests/wp21/README.md、REAL_WORKFLOW_PROTOCOL.md、EVIDENCE.md 与 REVIEW.md，核对基线SHA和包校验清单。按实际用户指令使用可信 manage.py 为每题生成独立工作树和外置control，不能在共享源码上同时运行20位agent。生成工作树时只有真实分派已经存在才填 assignment-ref，原文或来源摘要需要可核对，不冒充人类审批。

环境依赖在工作树外准备一次，共享模型/SAE和输入缓存只读；预检、冻结实际HF缓存/revision/文件hash并运行真实加载probe。CPU准备检查与模型正式运行分开记录。缺资源报告阻塞；已具备的条件继续推进。GPU同设备默认同时1个模型作业，CPU最多4个。每位agent收到自己的Rxx提示与对应workspace，只处理其允许路径。

先让agent写预期diff与必要负例，再完成真实生产代码修改和小规模实验。五道删除题用各自seed，不能等其他agent先完成新增。不给考生修改harness、case、fixture、评分器或正式Tree的权限。

收集完整自测、错误、命令退出码、实际产物与源码差异。从可信工具副本运行collect/audit；EVIDENCE_READY_FOR_REPLAY只是结构一致。由独立reviewer在另一工作树重新运行公开正反例、检查真实权重/数值、评定100分四项和硬门槛。每题保留原始结果，报告产品既有缺陷、agent错误、规则缺口、环境阻塞或评审范围问题，规则改善由维护者处理。

全程本地，交付20题状态表和原始证据。不push、不创建PR、不发布网站，不伪造人类批准。若用户只分派部分题，按该部分执行。
