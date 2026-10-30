"""R07 initial dataset witness. Load through DatasetPlugin.setup, then real SAEBase capture."""


from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only', action='store_true', help='Inspect registration/configuration without model execution')
    p.add_argument('--output', type=Path, help='Must equal this candidate workspace root; files stay under its experiment/ hierarchy')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--prompt', default='Mira placed a copper key in the blue box. The copper key is in the')
    return p


def configuration(args, task_file, *, dataset_file='conf/dataset/ioi.yaml', model_file='conf/model/openresearch_gemma2_2b_base.yaml'):
    from omegaconf import OmegaConf
    parent = (args.output or ROOT).resolve()
    if parent != ROOT.resolve():
        raise ValueError('--output must equal this candidate workspace root; artifacts belong under candidate/experiment')
    cfg = OmegaConf.merge(
        OmegaConf.create({'paths': {'root': str(parent), 'data_dir': str(parent / 'data')},
                          'experiment': {'sub_name': 'before_R07', 'overwrite': False},
                          'seed': 42, 'train': {'max_epochs': 1, 'facets': {'clear_experiment_leaf': True}}}),
        OmegaConf.load(ROOT / dataset_file),
        OmegaConf.load(ROOT / model_file),
        OmegaConf.load(ROOT / task_file),
    )
    cfg.task.prompt = args.prompt
    cfg.task.device = args.device
    cfg.task.execute = not args.check_only
    cfg.model.device = args.device
    return cfg


def new_run(cfg):
    from saebase.runtime.experiment_paths import run_leaf, prepare_run
    folder = run_leaf(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    prepare_run(cfg, folder)
    return folder


def write_result(folder, cfg, payload, arrays=None):
    from omegaconf import OmegaConf
    import numpy as np
    manifest = {}
    for name, value in (arrays or {}).items():
        if not name.replace('_', '').isalnum():
            raise ValueError('Unsafe array label')
        target = folder / 'metrics' / (name + '.npy')
        with target.open('xb') as handle:
            np.save(handle, value, allow_pickle=False)
        manifest[target.relative_to(folder).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    with (folder / 'metadata' / 'seed_result.json').open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    with (folder / 'metadata' / 'seed_resolved_config.json').open('x', encoding='utf-8') as handle:
        json.dump(OmegaConf.to_container(cfg, resolve=True), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    with (folder / 'metadata' / 'seed_array_hashes.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'status': 'executed', 'measured': True, 'run': str(folder), 'payload': payload}, sort_keys=True))


def check_result(details):
    print(json.dumps({'status': 'registration_only', 'measured': False, 'details': details}, sort_keys=True))

def main():
    p = parser()
    p.add_argument('--dataset-file', type=Path, default=ROOT / 'harness/tests/wp21/fixtures/common/prompts.jsonl')
    p.add_argument('--max-samples', type=int, default=2)
    args = p.parse_args()
    if not 1 <= args.max_samples <= 4:
        p.error('--max-samples must be between 1 and 4')
    from saebase.datasets.wp21_retired_prompts import KEY, ALIAS, register
    from saebase.registry import get_dataset_plugin, resolve_dataset_name
    from types import SimpleNamespace
    register()
    cfg = configuration(args, 'conf/task/capture.yaml', dataset_file='conf/dataset/wp21_retired_prompts.yaml')
    cfg.dataset.prompt_file = str(args.dataset_file.resolve())
    plugin = get_dataset_plugin(KEY)
    assert get_dataset_plugin(ALIAS) is plugin and resolve_dataset_name(ALIAS) == KEY
    plugin.assert_task_supported(cfg)
    dm = SimpleNamespace()
    plugin.setup(dm, cfg, 'test')
    split_counts = {'train': len(dm.train_set), 'validation': len(dm.val_set), 'test': len(dm.test_set)}
    if args.check_only:
        check_result({'dataset': KEY, 'alias': ALIAS, 'split_counts': split_counts})
        return
    if args.output is None:
        p.error('--output is required for a real run')
    from saebase.api import load_registered, capture_features
    from saebase.store import sha256_state_dict
    folder = new_run(cfg)
    pair = load_registered(model_key='gemma2_2b_base', sae_key='gemma2_2b_pt_l25_canonical', layer=25, device=args.device)
    records = dm.test_set[:args.max_samples]
    arrays = {}
    measurements = []
    for index, row in enumerate(records):
        snap = capture_features(pair.model, pair.sae, site=pair.site, prompt=row['prompt'])
        arrays[f'activation_{index}'] = snap.activation.detach().float().cpu().numpy()
        arrays[f'codes_{index}'] = snap.codes.detach().float().cpu().numpy()
        arrays[f'token_ids_{index}'] = snap.tokens.detach().cpu().numpy()
        measurements.append({'sample_id': row['sample_id'], 'prompt_group_id': row['prompt_group_id'],
                             'split': row['split'], 'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(),
                             'active_count': int((snap.codes != 0).sum().item())})
    payload = {'status': 'executed', 'operation': 'capture', 'dataset_key': KEY,
               'dataset_sha256': hashlib.sha256(args.dataset_file.read_bytes()).hexdigest(),
               'model_key': 'gemma2_2b_base', 'sae_key': 'gemma2_2b_pt_l25_canonical',
               'model_state_sha256': sha256_state_dict(pair.model.model),
               'sae_state_sha256': sha256_state_dict(pair.sae), 'measurements': measurements,
               'split_counts': split_counts}
    write_result(folder, cfg, payload, arrays)

if __name__ == '__main__':
    main()
