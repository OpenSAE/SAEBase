# `saebase.generate`

全仓只有一个 Python 环境。普通生成和改层都用它，不再拆 `.venv-vllm` 和 `.venv-nnsight-vllm`。

启动参数（端口、显存比例、`max_model_len`、dtype、FlashInfer）只在 `saebase/generate/settings.py`。

```text
task
  │  prompts + 采样参数
  ▼
saebase.generate.generate()          当前解释器里的 vLLM
saebase.generate.nnsight_worker      改层。python -m saebase.generate.nnsight_worker
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
from saebase.generate import generate

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
python -m saebase.generate.nnsight_worker smoke --model-id /path/to/weights --layer 0
```

`generate` 协议模式依赖原研究工作区尚未随 SAEBase 发布的 task 实现，因此当前独立版本只开放相应预检与基础 worker 能力。详见 [`harness/github_development_doc/PHASE_C.md`](../../harness/github_development_doc/PHASE_C.md)。
