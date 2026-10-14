"""task × prompt_source compatibility for ``gemma2_9b_it``."""

from __future__ import annotations

from typing import FrozenSet

from omegaconf import DictConfig

from saebase.data.prompt_schema import behaviors_for_model
from saebase.cfg.contracts import assert_prompt_source_in, assert_task_in

_TASKS: FrozenSet[str] = frozenset(
    {
        "sae",
        "ordered_steering",
        "pair_interaction",
        "vllm_sae_steer",
        "feature_filtering",
        "cross_model_filtering",
        "sae_upgrade",
        "targeted_replication",
        "pair_scale",
        "middepth_sae",
        "cross_model_transfer",
        "caa_steering",
        "caa_sample_size",
        "caa_feature_augmentation",
        "sample_caa",
        "caa_norm_matching",
        "caa_alpha_sweep",
        "feature_alpha_sweep",
        "feature_count_sweep",
        "functional_concept_validation",
        "whitened_localization_benchmark",
        "concept_judge_screening",
        "v_localized_interaction",
        "13d_interaction_metric_bakeoff",
        "13e_contextual_association",
        "13f_relative_white_i",
        "13g_background_normalized_i",
        "13h_output_value_interaction",
        "13i_pair_value_benchmark",
        "13j_empty_context_value",
        "13k_question_level_interaction",
        "13l_model_pooled_embedding",
        "13n_token_feature_association",
        "13o_peap_task_edges",
        "concept_guided_search",
        "14a_semantic_gating",
        "14b_prompt_use",
        "15_large_scale_validation",
    }
)

_ALIAS = {
    "CounterFact": "counterfact",
    "IOI": "ioi",
    "GreaterThan": "greater_than",
    "SubjectVerbAgreement": "subject_verb_agreement",
    "SimpleCopying": "simple_copying",
    "MMLUPro": "mmlu_pro",
    "BBH": "bbh",
    "ARCChallenge": "arc_challenge",
    "MedQA": "medqa",
    "MATH500": "math500",
    "GPQAExtended": "gpqa_extended",
    "LiveCodeBench": "livecodebench",
}


def assert_compatible(cfg: DictConfig) -> None:
    assert_task_in(cfg, _TASKS, "gemma2_9b_it")
    allowed = set(behaviors_for_model("gemma2_9b_it"))
    for beh in list(allowed):
        if beh in _ALIAS:
            allowed.add(_ALIAS[beh])
    assert_prompt_source_in(cfg, frozenset(allowed), "gemma2_9b_it")
