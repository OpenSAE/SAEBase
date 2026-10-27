# Environment Tree

The Environment Tree records the machine where an agent runs research. It uses observed facts for CPU, GPU, memory, storage, Python, terminal tools, and execution limitations rather than guesses.

```bash
python harness/environment/probe.py
python harness/environment/probe.py --output harness/environment/out/machine.json
python harness/environment/compare.py harness/environment/out/old.json harness/environment/out/new.json
```

Machine snapshots under `harness/environment/out/` are local artifacts and are not committed to Git. The public `harness/environment/example.json` is explicitly synthetic and must retain `synthetic: true`. If `nvidia-smi` is unavailable, the GPU state is **unknown**, not a claim that no GPU is present.

Ordinary training runs in the foreground. Background use of tmux or a cluster scheduler such as sbatch requires explicit user direction and actual support in the environment. Shell, network, and sandbox capabilities must be grounded in observed tool behavior.

Knowledge nodes may reference `environment_refs` to indicate required execution resources. An environment measurement is not scientific proof. A Code Tree cannot infer which GPU executed a method.

The public tree groups observed or synthetic environment properties by operating system, processor, memory, graphics devices, storage, and tools. Node titles use descriptive terms rather than commands or raw JSON.
