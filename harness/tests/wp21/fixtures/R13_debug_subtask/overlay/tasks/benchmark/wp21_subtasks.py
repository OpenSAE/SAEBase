"""Two explicit benchmark subtasks, each computed from a real SAE snapshot."""
from __future__ import annotations


def _debug_norm(snapshot):
    return {'metric': 'debug_activation_l2', 'value': float(snapshot.activation.detach().float().norm().item())}


def _reconstruction(snapshot):
    difference = snapshot.reconstruction.detach().float() - snapshot.activation.detach().float()
    return {'metric': 'reconstruction_mse', 'value': float(difference.square().mean().item())}


SUBTASKS = {'debug_norm': _debug_norm, 'reconstruction': _reconstruction}


def list_subtasks():
    return tuple(sorted(SUBTASKS))


def execute(cfg):
    from tasks.sae_runtime import _task, _pair, _prompt, _fingerprint_prompt
    selector = _task(cfg, 'subtask')
    if selector not in SUBTASKS:
        raise ValueError(f'Unknown benchmark subtask: {selector!r}')
    if _task(cfg, 'execute') is not True:
        raise ValueError('Subtask measurements require task.execute=true')
    from saebase.api import capture_features
    from saebase.store import sha256_state_dict
    from pathlib import Path
    import hashlib
    import numpy as np
    pair = _pair(cfg)
    prompt = _prompt(cfg)
    snapshot = capture_features(pair.model, pair.sae, site=pair.site, prompt=prompt)
    values = {'activation': snapshot.activation.detach().float().cpu().numpy(),
              'codes': snapshot.codes.detach().float().cpu().numpy(),
              'reconstruction': snapshot.reconstruction.detach().float().cpu().numpy(),
              'token_ids': snapshot.tokens.detach().cpu().numpy()}
    raw = {}
    folder = Path.cwd() / 'metrics'
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        target = folder / ('subtask_' + name + '.npy')
        with target.open('xb') as output:
            np.save(output, value, allow_pickle=False)
        raw[target.relative_to(Path.cwd()).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    return {'status': 'executed', 'operation': 'benchmark', 'subtask': selector,
            'result': SUBTASKS[selector](snapshot), 'raw_sha256': raw,
            'model_key': pair.identity.model_key, 'sae_key': pair.identity.sae_key,
            'hook_name': pair.site.hook_name, 'prompt_sha256': _fingerprint_prompt(prompt),
            'model_state_sha256': sha256_state_dict(pair.model.model),
            'sae_state_sha256': sha256_state_dict(pair.sae)}
