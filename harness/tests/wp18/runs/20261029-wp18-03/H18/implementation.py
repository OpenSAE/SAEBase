"""Match a synthetic SAE hook to its declared model, width, and layer."""

from __future__ import annotations


class InputError(ValueError):
    """Raised when model identity, width, or layer does not match the SAE."""


def solve(payload: dict) -> dict:
    """Return the matched model id, SAE id, and layer. Missing fields stay missing."""
    if not isinstance(payload, dict) or set(payload) != {"model", "sae"}:
        raise InputError("schema")
    model = _model(payload["model"])
    sae = _sae(payload["sae"])
    if sae["model_id"] != model["id"] or sae["d_in"] != model["hidden_size"]:
        raise InputError("identity")
    if sae["layer"] not in model["layers"]:
        raise InputError("layer")
    return {"model": model["id"], "sae": sae["id"], "layer": sae["layer"]}


def _model(value: object) -> dict[str, object]:
    """Read id, hidden_size, and the explicit layer list."""
    if not isinstance(value, dict) or set(value) != {"id", "hidden_size", "layers"}:
        raise InputError("model")
    model_id = value["id"]
    width = value["hidden_size"]
    layers = value["layers"]
    if not isinstance(model_id, str) or not model_id:
        raise InputError("model")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise InputError("model")
    if not isinstance(layers, list) or not layers:
        raise InputError("model")
    clean: list[int] = []
    for layer in layers:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer in clean:
            raise InputError("model")
        clean.append(layer)
    return {"id": model_id, "hidden_size": width, "layers": clean}


def _sae(value: object) -> dict[str, object]:
    """Read SAE id, model_id, d_in, and layer without filling defaults."""
    if not isinstance(value, dict) or set(value) != {"id", "model_id", "d_in", "layer"}:
        raise InputError("sae")
    sae_id = value["id"]
    model_id = value["model_id"]
    width = value["d_in"]
    layer = value["layer"]
    if not isinstance(sae_id, str) or not sae_id or not isinstance(model_id, str) or not model_id:
        raise InputError("sae")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise InputError("sae")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise InputError("sae")
    return {"id": sae_id, "model_id": model_id, "d_in": width, "layer": layer}
