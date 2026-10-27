"""Release only SAEBase-owned generator resources.

External vLLM instances are never terminated by SAEBase cleanup.
"""

from __future__ import annotations

import gc
import time


def stop_vllm() -> None:
    """Release in-process vLLM and the HTTP server started by THIS process.

    Does not search global processes or terminate externally launched servers.
    An externally configured VLLM_BASE_URL remains untouched.
    """
    from tasks.LLM.generate.backend.engine import release
    from tasks.LLM.generate.backend.server import stop_owned_server

    release()
    stop_owned_server()


def free_cuda() -> None:
    """Release PyTorch CUDA caches in this interpreter."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    time.sleep(2.0)
