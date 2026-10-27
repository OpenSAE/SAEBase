# Script Guide

Use the published `saebase-*` commands or [task-based examples](../saebase/examples/README.md) for ordinary research. The scripts in this folder support the same workflows and preserve commands used to reproduce earlier results.

| Purpose | Entry points |
| --- | --- |
| Benchmark and feature-edit CLI implementation | [Benchmark](m3_benchmark.py) · [Interventions](m2_batch.py) · [Reliability](m4_reliability.py) |
| Model compatibility and packaging | [Environment](check_env.py) · [Documentation links](check_documentation.py) · [Distribution](verify_distribution.py) |
| Current GPU performance evidence | [Qwen intervention benchmark](workplan11_acceptance_gpu.py) · [KV-cached steering](workplan11_fast_steering_gpu.py) |
| Earlier experiment reproduction | [Performance measurements](workplan09_gpu_pilot.py) · [Model coverage records](phase_d_verify_evidence.py) |

Scripts with historical stage or plan numbers are reproducibility entrypoints; they are not sequential prerequisites for a new SAEBase installation. The [repository guide](../harness/github_development_doc/repository-guide.md) explains where source code, examples, configurations and evidence belong.
