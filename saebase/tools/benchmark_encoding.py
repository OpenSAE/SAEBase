"""Offline, repeatable SAE encoding microbenchmark; not a GPU claim.

Run: python saebase/tools/benchmark_encoding.py --rows 8192 --width 1024 --repeat 5
Compare batch=1 against larger batches using the same fixed tensors.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import torch


class FixedSAE(torch.nn.Module):
    def __init__(self, width: int, features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(width, features) / width**0.5, requires_grad=False
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.weight)


def measure(model: FixedSAE, rows: torch.Tensor, chunk: int, repeat: int) -> dict:
    times = []
    device = rows.device
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = perf_counter()
        with torch.inference_mode():
            for block in rows.split(chunk):
                model.encode(block)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(perf_counter() - started)
    times.sort()
    median = times[len(times) // 2]
    return {
        "batch_rows": chunk,
        "median_seconds": median,
        "rows_per_second": round(len(rows) / median, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-sizes", default="1,16,64,256")
    args = parser.parse_args()
    if min(args.rows, args.width, args.features, args.repeat) < 1:
        parser.error("rows, width, features and repeat must all be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(7)
    device = torch.device(args.device)
    model = FixedSAE(args.width, args.features).to(device).eval()
    x = torch.randn(args.rows, args.width, device=device)
    sizes = [int(x) for x in args.batch_sizes.split(",")]
    if any(x < 1 for x in sizes):
        parser.error("batch sizes must be positive")
    result = {
        "device": args.device,
        "rows": args.rows,
        "width": args.width,
        "features": args.features,
        "torch": torch.__version__,
        "measurements": [measure(model, x, n, args.repeat) for n in sizes],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
