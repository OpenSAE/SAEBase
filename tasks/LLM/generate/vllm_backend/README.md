# `saebase/vllm_backend`（兼容层）

实现已迁到 [`saebase/generate/`](../../saebase/generate/README.md)。本包只 re-export，并带上 04 采样默认值（temp 0.6 / top_p 0.95 / max_new 4096）和 8B 本地路径。

新代码：

```python
from saebase.generate import generate
```

选型、吞吐数字、5090 坑：[`doc/04_chain_guidance/04a_hf_generate_speedup.md`](../../doc/04_chain_guidance/04a_hf_generate_speedup.md)。
