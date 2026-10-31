"""Model-only extension around the existing Gemma factory, for retirement.

Provider revision hashes this adapter source. It is explicitly separate from
the model checkpoint state hash measured only after the actual model loads.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

KEY = 'ext.wp21.retired_gemma'
BUNDLE = 'ext.wp21.retired_gemma_bundle'


def build_bundle(expected_model_sha256=None):
    from saebase.api import ActivationSite
    from saebase.extensions.contracts import ExtensionBundle, ExternalModel, HookSpec
    if expected_model_sha256 is not None:
        if len(expected_model_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_model_sha256):
            raise ValueError('Expected lowercase SHA-256 for the actual loaded model state')
    def factory(device):
        from models.gemma2_2b_base.gemma2_2b_base import load_model
        from saebase.store import sha256_state_dict
        handle = load_model(device=device)
        if expected_model_sha256 is not None and sha256_state_dict(handle.model) != expected_model_sha256:
            raise ValueError('Loaded Gemma state differs from the frozen provisioned checkpoint')
        return handle.model
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provider = ExternalModel(key=KEY, source='local-provider:models/wp21_retired_gemma.py',
                             revision='sha256:' + source_sha,
                             hooks=(HookSpec(ActivationSite(25, 'blocks.25.hook_resid_post'), 2304),),
                             factory=factory, backend='auto')
    return ExtensionBundle(name=BUNDLE, models=(provider,))


def register(expected_model_sha256=None):
    from saebase.extensions.registry import register_extension
    register_extension(build_bundle(expected_model_sha256))
