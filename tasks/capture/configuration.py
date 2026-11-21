"""Read capture settings by owner and translate saved pre-migration task settings."""

_LEGACY_SAR_KEYS = {
    "output": (),
    "g_s": (),
    "selection": ("feature_filter", "cohort_filter", "feature_selection"),
    "answer": ("answer_backend", "answer"),
    "i": ("i_feature_selection", "i_g_s_shortfall", "max_exhaustive_features"),
    "sar": (
        "algorithm",
        "source",
        "top_pairs",
        "pair_ranking",
        "M",
        "n_fit",
        "p",
        "readout",
        "readout_position",
        "target_token_id",
        "reference_token_id",
        "require_affine_decode",
    ),
}
_LEGACY_NAMES = {
    "answer_backend": "backend",
    "answer": "reference_text",
    "i_feature_selection": "feature_selection",
    "i_g_s_shortfall": "g_s_shortfall",
}


def capture_section(task, name):
    """Resolve one section, with the current keys taking precedence over saved aliases."""
    capture = task.get("capture", {})
    if not hasattr(capture, "get"):
        raise ValueError("task.capture must contain capture settings")
    current = capture.get(name, {})
    if not hasattr(current, "items"):
        raise ValueError(f"task.capture.{name} must contain settings")
    if name == "rss":
        previous = task.get("rss", {})
        if not hasattr(previous, "items"):
            raise ValueError("task.capture.rss must contain settings")
        previous = dict(previous)
        if "dominance_ratio" in previous:
            previous["report"] = {"dominance_ratio": previous.pop("dominance_ratio")}
    else:
        sar = task.get("sar", {})
        if not hasattr(sar, "get"):
            raise ValueError("task.capture.sar must contain SAR settings")
        previous = {
            _LEGACY_NAMES.get(key, key): sar[key] for key in _LEGACY_SAR_KEYS[name] if key in sar
        }
    if name == "selection" and "filter" in task:
        previous["candidate_count"] = task["filter"]
    if name == "output":
        count = task.get("capture_count")
        if count is None:
            count = task.get("top_feature_num")
        if count is not None or "capture_count" in task:
            previous["feature_count"] = count
    return {**previous, **current}


def migrate_capture_task(task):
    """Resolve old overrides before merging defaults or comparing run configurations."""
    task = dict(task)
    sections = dict(task.get("capture", {}))
    for key in (
        "mode",
        "sample_ids",
        "output_scope",
        "max_tokens",
        "max_active_per_token",
        "whiten_chunk_size",
        "neuronpedia",
    ):
        if key in task:
            sections.setdefault(key, task.pop(key))
    for name in (*_LEGACY_SAR_KEYS, "rss"):
        settings = capture_section(task, name)
        if settings:
            sections[name] = settings
    task.pop("sar", None)
    task.pop("rss", None)
    task.pop("filter", None)
    task.pop("capture_count", None)
    task.pop("top_feature_num", None)
    if sections:
        task["capture"] = sections
    return task
