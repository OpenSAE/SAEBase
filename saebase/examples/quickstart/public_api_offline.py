"""Standalone SAEBase API walkthrough; needs only torch and local project code.

Run: python saebase/examples/quickstart/public_api_offline.py
Shows selection of a particular token/site, SAE editing and output effects.
Numbers are from an intentionally tiny synthetic model/SAE, not paper results.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from saebase.api import (
    ActivationSite, ModelAdapter, SAEIdentity, capture_features,
    coalition_replacements, edit_features, evaluate_edit,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.output.weight.copy_(torch.tensor([
                [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.2, .1, -.2]
            ]))
        self.cfg = type("Config", (), {"n_layers": 1, "device": "cpu"})()

    def to_tokens(self, prompt: str):
        return torch.tensor([[int(v) for v in prompt.split()]], dtype=torch.long)

    def activation(self, tokens: torch.Tensor):
        x = tokens.float().unsqueeze(-1)
        return torch.cat([x / 3, x / 5, x / 7], dim=-1)

    def forward(self, tokens):
        return self.output(torch.cumsum(self.activation(tokens), dim=1))

    def run_with_cache(self, tokens, names_filter=None):
        return self(tokens), {"blocks.0.hook_resid_post": self.activation(tokens)}

    def run_with_hooks(self, tokens, fwd_hooks):
        h = self.activation(tokens)
        for hook_name, callback in fwd_hooks:
            assert hook_name == "blocks.0.hook_resid_post"
            h = callback(h, None)
        return self.output(torch.cumsum(h, dim=1))


class TinySAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = type("Config", (), {"d_in": 3, "d_sae": 4})()
        self.bias = torch.nn.Parameter(torch.tensor([.01, -.02, .04]))

    def encode(self, x):
        return torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)

    def decode(self, z):
        return z[..., :3] * .8 + self.bias


def main():
    model = ModelAdapter(TinyModel().eval())
    sae = TinySAE().eval()
    site = ActivationSite(layer=0, hook_name="blocks.0.hook_resid_post", position=-2)
    identity = SAEIdentity.for_custom(
        sae=sae, model_key="tiny", checkpoint="local-synthetic-demo",
        hook_name=site.hook_name, layer=site.layer,
    )
    snapshot = capture_features(model, sae, site=site, prompt="3 5 7")
    replacements = coalition_replacements(
        snapshot.codes, selected_indices=[0, 1], retained_indices=[1]
    )
    patch = edit_features(sae, snapshot, replacements=replacements)
    result = evaluate_edit(model, sae, site=site, prompt="3 5 7", replacements=replacements)
    assert torch.allclose(patch.activation - snapshot.residual,
                          sae.decode(patch.codes[None])[0])
    assert not torch.allclose(result.baseline_logits, result.edited_logits)
    record = {
        "site": {"hook_name": site.hook_name, "layer": site.layer,
                 "token_position": site.position},
        "sae_identity": identity.to_dict(),
        "baseline_logits": result.baseline_logits.tolist(),
        "edited_logits": result.edited_logits.tolist(),
        "logit_delta": result.logit_delta.tolist(),
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
