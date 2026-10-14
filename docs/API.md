# SAEBase API at a glance

The public interface resides in `saebase.api`.

- `ModelAdapter`: standardizes model activation capture and patch execution.
- `ActivationSite`: identifies layer, named hook, and token position.
- `SAEIdentity`: records checkpoint/model/hook/layer information.
- `capture_features`: returns raw activations, SAE codes and residuals.
- `edit_features`: performs `h' = h + decode(z') - decode(z)`.
- `evaluate_edit`: compares logits with and without a patch and checks identity replay.

Run `python examples/05_public_api_offline.py` for a self-contained API example.
Run `python examples/06_public_api_pythia.py` for an opt-in public-checkpoint example.

## Integrations

- `saebase.integrations.sar`: supports coalition-based feature edits while
  calling the original `sar.py` in a separate checkout.
- `saebase.integrations.rss`: captures clean/selection/held-out SAE codes and
  calls the original `rss.py` in a separate checkout.

See `docs/PHASE_B.md` and `docs/PHASE_C.md` for the full contract.
