# `tasks.generate.backend`

## 目的和主要功能

提供基线生成、vLLM 调用及 NNsight 工作进程能力，供生成任务与指定调用者使用。

## 适用范围

用于调用或维护生成后端。运行采用所选解释器和实际后端依赖，采样条件由调用者提供。

启动参数（端口、显存比例、`max_model_len`、dtype、FlashInfer）只在 `tasks/generate/backend/settings.py`。

```text
task
  │  prompts + 采样参数
  ▼
tasks.generate.backend.generate()          当前解释器里的 vLLM
tasks.generate.backend.nnsight_worker      改层。python -m tasks.generate.backend.nnsight_worker
```

HuggingFace packed decode 只有调用方显式 `SAE_GENERATE_BACKEND=hf` 时，自己去调 `generate_hf_packed`。`generate()` 在这个开关下会抛错。

## 安装

标准 SAE 分析依赖在根目录 `requirements.txt`。GPU 生成和 NNsight 通过项目的可选依赖 `generation` 安装：

```bash
python -m pip install -e '.[generation]'
```

具体的 CUDA、PyTorch 与 vLLM 版本必须在目标机器上验证。`apply_process_env()` 设置项目默认环境变量，可由调用者覆盖。

## 调用

```python
from tasks.generate.backend import generate

rows = generate(
    prompts,
    max_new_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    model="/path/to/weights",
    seed=0,  # None 时不传 seed
)
```

采样参数由调用方传入。这里不写模型名，也不写协议默认值。

当前解释器能 `import vllm` 就在进程里生成。已经有人在 `VLLM_BASE_URL` 上把服务开好了，才走 HTTP。不会再去拉起另一个虚拟环境。

改层的独立 GPU 验收入口：

```bash
python -m tasks.generate.backend.nnsight_worker smoke --model-id /path/to/weights --layer 0
```

基线生成、层探测和激活导出由当前工作进程提供；特征编辑使用当前 generate/intervene 接口。新增调用先明确生成文本或激活将用于哪项工作，再选择返回、保存和读取方式。后端失败保留实际对象与错误信息，调用者据此处理具体资源或条件；正式任务的完成判断与结果记录由所属 Task 承担。

[当前架构](../../../doc/code/development/current-architecture.md)说明 Task、公共接口与运行支持的职责。
