import torch

from saebase.interventions.steering import _distribution_shift


def test_distribution_shift_reports_rank_and_probability():
    clean = torch.tensor([0.0, 3.0, 1.0])
    edited = torch.tensor([0.0, 1.0, 4.0])
    shift = _distribution_shift(clean, edited, feature_id=7, activation=1.5)
    assert shift["clean_argmax_token_id"] == 1
    assert shift["edited_argmax_token_id"] == 2
    assert shift["clean_argmax_rank_after_edit"] == 1
    assert shift["feature_id"] == 7
    assert shift["max_abs_logit_delta"] == 3.0
    assert shift["logit_kind"] == "teacher_forced_next_token_at_generation_step"


def test_identical_logits_have_zero_shift():
    logits = torch.tensor([1.0, 2.0, 0.5])
    shift = _distribution_shift(logits, logits.clone(), feature_id=None, activation=None)
    assert shift["max_abs_logit_delta"] == 0.0
    assert shift["clean_argmax_rank_after_edit"] == 0
    assert shift["edited_argmax_token_id"] == shift["clean_argmax_token_id"]
