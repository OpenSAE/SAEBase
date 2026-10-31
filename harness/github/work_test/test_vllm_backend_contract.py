"""Tests for import-safe standalone legacy wrapper; no protocol-04 code bundled."""
import pytest
from saebase.vllm_backend.generate import packed_row, sampling_params_kwargs, generate_vllm


def test_packed_row_has_expected_fields():
    row=packed_row(prompt="hi", token_ids=[1,2,3,4], prompt_len=2, text="ab")
    assert row["n_gen_tokens"]==2
    assert row["token_ids"]==[1,2,3,4]


def test_sampling_requires_explicit_protocol():
    with pytest.raises(TypeError):
        sampling_params_kwargs()
    assert sampling_params_kwargs(max_new_tokens=120,temperature=0.2,top_p=0.9)=={
        "max_tokens":120,"temperature":0.2,"top_p":0.9
    }


def test_generation_wrapper_propagates_settings(monkeypatch):
    import saebase.vllm_backend.generate as module
    saved=[]
    monkeypatch.setattr(module,"_generate_vllm",lambda prompts,**kwargs: saved.append((prompts,kwargs)) or [])
    assert generate_vllm(["hi"],max_new_tokens=10,temperature=0.1,top_p=0.95,model="test") == []
    assert saved[0][0] == ["hi"]
    assert saved[0][1]["max_new_tokens"] == 10
