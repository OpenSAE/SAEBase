"""Explicitly registered WP21 deletion fixture with its own strict prompt reader.

The generic built-in prompt reader has a closed behavior list. This provider
owns its reader; no built-in matrix or private registry is patched.
"""
from __future__ import annotations
import json
from pathlib import Path

KEY = 'WP21RetiredPrompts'
ALIAS = 'wp21_retired_prompts'


def read_records(path):
    from saebase.datasets.prompt_schema import require_keys
    required = ['sample_id', 'prompt_group_id', 'split', 'prompt']
    rows = []
    ids = set()
    group_splits = {}
    with Path(path).open(encoding='utf-8') as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'Expected object at line {line_no}')
            require_keys(row, required, ctx=f'{KEY}:{line_no}')
            if set(row) != set(required) or any(not isinstance(row[x], str) or not row[x].strip() for x in required):
                raise ValueError('Exact nonempty string fields required')
            if row['split'] not in {'train', 'validation', 'test'}:
                raise ValueError('Unknown explicit split')
            if row['sample_id'] in ids:
                raise ValueError('Duplicate sample ID')
            ids.add(row['sample_id'])
            previous = group_splits.setdefault(row['prompt_group_id'], row['split'])
            if previous != row['split']:
                raise ValueError('Prompt group crosses splits')
            rows.append(row)
    if not rows:
        raise ValueError('Empty prompt file')
    return rows


def setup(dm, cfg, stage=None):
    records = read_records(cfg.dataset.prompt_file)
    dm.train_set = [r for r in records if r['split'] == 'train']
    dm.val_set = [r for r in records if r['split'] == 'validation']
    dm.test_set = [r for r in records if r['split'] == 'test']


def assert_task_supported(cfg):
    if str(cfg.dataset.name) not in {KEY, ALIAS} or str(cfg.task.name) != 'capture':
        raise ValueError('This fixture dataset supports only its explicit capture experiment')
    if str(cfg.model.name) != 'gemma2_2b_base':
        raise ValueError('This fixture uses the canonical Gemma-2 2B backbone')


def split_note(dm):
    return 'Explicit file splits only; group IDs stay within one split.'


def register():
    from saebase.plugin_bootstrap import ensure_plugins_loaded
    from saebase.registry import register_dataset, register_dataset_alias
    ensure_plugins_loaded()
    register_dataset(KEY, setup, assert_task_supported, split_note)
    register_dataset_alias(ALIAS, KEY)
