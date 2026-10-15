"""Reconstruction behavioral NLL and loss recovered with explicit ablation reference."""
from __future__ import annotations
import math
import torch
from saebase.api import ModelAdapter, ActivationSite
from saebase.interventions import MeasurementSample

@torch.no_grad()
def target_nll(model: ModelAdapter, sae, sample: MeasurementSample, site: ActivationSite):
    """Only a single-token NLL at the explicitly requested readout, not whole-text CE."""
    if sample.target_token_id is None:
        raise ValueError("target_nll requires target_token_id")
    ids = torch.tensor([sample.tokens],dtype=torch.long)
    baseline = model.forward_logits(tokens=ids, readout_position=sample.readout_position)
    # M2 reconstruction is full h -> decode(encode(h)), not residual-preserving.
    h = model.capture(tokens=ids,site=ActivationSite(site.layer,site.hook_name,sample.patch_position))
    try:
        param = next(sae.parameters())
        inp = h.to(device=param.device,dtype=param.dtype)
    except (AttributeError,StopIteration,TypeError):
        inp = h
    decoded = sae.decode(sae.encode(inp.unsqueeze(0)))[0].to(device=h.device,dtype=h.dtype)
    patched = model.patch_logits(tokens=ids,site=ActivationSite(site.layer,site.hook_name,sample.patch_position),
                                 activation=decoded,readout_position=sample.readout_position)
    t = sample.target_token_id
    if t >= len(baseline):
        raise ValueError("target token outside output vocab")
    # Full-site zero ablation is measured in the same model and at the same
    # token position, rather than trusting an unrelated user-entered scalar.
    ablated = model.patch_logits(tokens=ids,
        site=ActivationSite(site.layer,site.hook_name,sample.patch_position),
        activation=torch.zeros_like(h), readout_position=sample.readout_position)
    return (float(-torch.log_softmax(baseline.float(),-1)[t].item()),
            float(-torch.log_softmax(patched.float(),-1)[t].item()),
            float(-torch.log_softmax(ablated.float(),-1)[t].item()))

def loss_recovered(clean_loss: float, reconstructed_loss: float, ablated_loss: float):
    """(L_ablated - L_reconstructed)/(L_ablated - L_original)."""
    if not all(math.isfinite(x) for x in (clean_loss,reconstructed_loss,ablated_loss)):
        raise ValueError("nonfinite NLL")
    denom = ablated_loss-clean_loss
    if abs(denom) <= 1e-12:
        return None
    return (ablated_loss-reconstructed_loss)/denom
