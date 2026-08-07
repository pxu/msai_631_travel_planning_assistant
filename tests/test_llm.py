"""Tests for model selection in ``travel_assistant.llm``.

Only the *resolution* logic is tested — which checkpoint id gets chosen for
which device and environment. Actually loading weights is what `evals/` is
for; nothing here downloads a model.
"""

import pytest

from travel_assistant import llm as llm_module


@pytest.fixture
def captured(monkeypatch):
    """Intercept the load so `build_chat_model` resolves an id and stops."""
    seen = {}

    class Stop(Exception):
        pass

    def fake_from_pretrained(model_id, *args, **kwargs):
        seen["model_id"] = model_id
        raise Stop

    monkeypatch.setattr(llm_module.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        llm_module.HuggingFacePipeline,
        "from_model_id",
        classmethod(lambda cls, model_id, **kw: fake_from_pretrained(model_id)),
    )
    seen["Stop"] = Stop
    return seen


def _resolve(captured, **kwargs):
    with pytest.raises(captured["Stop"]):
        llm_module.build_chat_model(**kwargs)
    return captured["model_id"]


def test_defaults_to_the_quantized_checkpoint_on_cuda(monkeypatch, captured):
    """bitsandbytes 4-bit weights need CUDA kernels, so the quantized
    checkpoint is CUDA-only."""
    monkeypatch.setattr(llm_module, "_pick_device", lambda: "cuda")
    assert _resolve(captured) == llm_module.DEFAULT_CUDA_MODEL_ID


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_defaults_to_unquantized_weights_off_cuda(monkeypatch, captured, device):
    monkeypatch.setattr(llm_module, "_pick_device", lambda: device)
    assert _resolve(captured) == llm_module.DEFAULT_LOCAL_MODEL_ID


def test_env_var_overrides_the_default(monkeypatch, captured):
    """Lets `evals/` be pointed at a candidate model without a code change,
    so "would a bigger model help?" is measurable rather than a guess."""
    monkeypatch.setattr(llm_module, "_pick_device", lambda: "mps")
    monkeypatch.setenv(llm_module.MODEL_ENV_VAR, "unsloth/Llama-3.2-3B-Instruct")
    assert _resolve(captured) == "unsloth/Llama-3.2-3B-Instruct"


def test_explicit_argument_beats_the_env_var(monkeypatch, captured):
    monkeypatch.setattr(llm_module, "_pick_device", lambda: "mps")
    monkeypatch.setenv(llm_module.MODEL_ENV_VAR, "from-env")
    assert _resolve(captured, model_id="from-argument") == "from-argument"


def test_empty_env_var_falls_back_to_the_default(monkeypatch, captured):
    monkeypatch.setattr(llm_module, "_pick_device", lambda: "mps")
    monkeypatch.setenv(llm_module.MODEL_ENV_VAR, "")
    assert _resolve(captured) == llm_module.DEFAULT_LOCAL_MODEL_ID
