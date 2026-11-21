"""Four-state searches must use one background and never cherry-pick repeats."""

import numpy as np

from tasks.capture.pair_search import search_patterns


def test_joint_gain_is_found_with_the_three_baselines_flat():
    masks = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    values = np.log([0.60, 0.61, 0.605, 0.90])
    result = search_patterns([10, 20, 30], masks, values)
    assert result["strict_match_count"] == 1
    best = result["best"]
    assert best["pair_ids"] == [10, 20]
    assert best["background"] == {"30": 1}
    assert best["interaction"] > 0
    assert abs(best["both_gain_over_highest_baseline"] - 0.29) < 1e-12


def test_states_from_different_backgrounds_cannot_form_a_pair():
    masks = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
    result = search_patterns([10, 20, 30], masks, np.log([0.60, 0.61, 0.605, 0.90]))
    assert result["complete_four_state_backgrounds"] == 0
    assert result["strict_match_count"] == 0


def test_exactly_fifty_percent_is_excluded():
    masks = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    result = search_patterns([10, 20], masks, np.log([0.5, 0.51, 0.51, 0.90]))
    assert result["baseline_qualified_count"] == 0


def test_duplicate_high_value_does_not_replace_first_observation():
    masks = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [1, 1]])
    result = search_patterns([10, 20], masks, np.log([0.60, 0.61, 0.605, 0.62, 0.99]), gain=0.05)
    assert result["strict_match_count"] == 0
    assert result["candidates"][0]["coalition_indices_zero_based"]["11"] == 3
    assert result["candidates"][0]["duplicate_condition_indices"]["11"] == [3, 4]


def test_default_does_not_impose_five_percentage_point_gain_floor():
    masks = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    result = search_patterns([10, 20], masks, np.log([0.60, 0.601, 0.602, 0.61]))
    assert result["strict_match_count"] == 1
    assert result["best"]["both_gain_over_highest_baseline"] < 0.05


def test_rank_by_logprob_interaction_not_probability_gain():
    masks = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    )
    probabilities = [0.51, 0.51, 0.51, 0.54, 0.91, 0.91, 0.91, 0.95]
    result = search_patterns([10, 20, 30], masks, np.log(probabilities))
    assert result["best"]["pair_ids"] == [10, 20]
    assert result["best"]["background"] == {"30": 0}
    assert result["best"]["both_gain_over_highest_baseline"] < 0.04
