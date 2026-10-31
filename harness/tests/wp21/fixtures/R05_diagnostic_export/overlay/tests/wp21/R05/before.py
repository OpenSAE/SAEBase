"""R05 initial feature witness. Real mode runs the optional diagnostic on Gemma."""


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
                          'experiment': {'sub_name': 'before_R05', 'overwrite': False},
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
    args = p.parse_args()
    from tasks.capture.wp21_full_codes import capture_with_full_codes
    from saebase.registry import get_task_plugin
    cfg = configuration(args, 'conf/task/wp21_full_codes.yaml')
    if args.check_only:
        assert callable(capture_with_full_codes)
        assert get_task_plugin('capture').task_class.__name__ == 'CaptureTask'
        check_result({'feature': 'wp21_full_codes', 'capture_preserved': True})
        return
    if args.output is None:
        p.error('--output is required for a real run')
    folder = new_run(cfg)
    payload, arrays = capture_with_full_codes(cfg)
    write_result(folder, cfg, payload, arrays)

if __name__ == '__main__':
    main()
