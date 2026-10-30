"""Seeded optional full-code diagnostic exporter, to be retired by R05."""
from __future__ import annotations


def capture_with_full_codes(cfg):
    from tasks.sae_runtime import _require_context, _pair, _prompt, _fingerprint_prompt
    from saebase.api import capture_features
    from saebase.store import sha256_state_dict
    import torch
    if cfg.task.get('export_full_codes') is not True or cfg.task.get('execute') is not True:
        raise ValueError('This diagnostic needs explicit execute and export_full_codes flags')
    _require_context(cfg, 'capture')
    pair = _pair(cfg)
    prompt = _prompt(cfg)
    snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=prompt)
    codes = snapshot.codes.detach().float().cpu()
    payload = {'status': 'executed', 'operation': 'capture', 'diagnostic': 'wp21_full_codes',
               'full_codes_exported': True, 'model_key': pair.identity.model_key,
               'sae_key': pair.identity.sae_key, 'hook_name': pair.site.hook_name,
               'prompt_sha256': _fingerprint_prompt(prompt), 'feature_count': codes.numel(),
               'active_count': torch.count_nonzero(codes).item(),
               'model_state_sha256': sha256_state_dict(pair.model.model),
               'sae_state_sha256': sha256_state_dict(pair.sae)}
    arrays = {'full_codes': codes.numpy(),
              'activation': snapshot.activation.detach().float().cpu().numpy(),
              'reconstruction': snapshot.reconstruction.detach().float().cpu().numpy(),
              'token_ids': snapshot.tokens.detach().cpu().numpy()}
    return payload, arrays
