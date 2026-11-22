# `tasks.generate.serving`

## Purpose

Provide baseline generation, VLLM call and NNsight workflow capacity for task generation and caller designation.

## Scope

Runs with the selected interpreter and the actual backend dependencies, and the sampling conditions are provided by the caller.

Start parameters (port, visible ratio, `max_model_len`, dtype, FlashInfer) only at `tasks/generate/serving/settings.py`.

```text
task
  │  prompts + Sample Parameters
  ▼
tasks.generate.serving.generate()          From Current Interpreter vLLM
tasks.generate.serving.nnsight_worker      Alternating. python -m tasks.generate.serving.nnsight_worker
```

HuggingFace packed decode Only caller displays. `SAE_GENERATE_BACKEND=hf` Do it yourself. `generate_hf_packed`.`generate()` It's gonna be a mistake at this switch..

## Install

Standard SAE analysis relies on `requirements.txt` in the root directory. GPU generation and NNsight is optional to install `generation`:

```bash
python -m pip install -e '.[generation]'
```

Specific CUDA, PyTorch and vLLM versions must be validated on the target machine. `apply_process_env()` Sets the default environment variable for the project to be covered by the caller.

## Call

```python
from tasks.generate.serving import generate

rows = generate(
    prompts,
    max_new_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    model="/path/to/weights",
    seed=0,  # None It's not passed. seed
)
```

Sample parameters are entered by the caller. No model name or protocol default is written here.

Current interpreter can `import vllm` It's in the process. Someone's already in the process. `VLLM_BASE_URL` The service is ready. I'm leaving. HTTP.I'm not going to pull up another virtual environment..

Independent GPU retrofit:

```bash
python -m tasks.generate.serving.nnsight_worker smoke --model-id /path/to/weights --layer 0
```

Baseline generation, layer detection and activation are provided by the current working process; feature editing uses the current generate/intervene interface. The new call is first to specify the work to be used to generate the text or activate it, and then to choose how to return, save and read. The backend fails to keep the actual object and the wrong information, on which the caller handles the specific resource or condition; the official task completion judgement and result record is assumed by the parent Task.

[Current structure](../../../doc/code/documentation/source/development/current-architecture.md)Annotations Task,Public interfaces and functions of operation support.
