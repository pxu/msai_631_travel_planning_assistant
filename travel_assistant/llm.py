"""LangChain-wrapped chat model used by the generation nodes.

Per the project proposal, inference runs locally on a free-tier GPU using
a small open-source instruct model, so no API key is required. This
module wraps that model with LangChain's HuggingFace integration
(``HuggingFacePipeline`` + ``ChatHuggingFace``) instead of calling
``transformers`` directly, so the rest of the app talks to a standard
LangChain chat model interface (``.invoke(messages) -> AIMessage``).

Two backends are supported:

- CUDA (e.g. Colab's free T4, as in the proposal): loads the proposal's
  pre-quantized ``unsloth/Llama-3.2-1B-Instruct-bnb-4bit`` checkpoint,
  since ``bitsandbytes`` 4-bit quantization requires CUDA kernels.
- Apple Silicon (MPS) or CPU: ``bitsandbytes`` has no Metal/CPU backend,
  so this loads the same model family's non-quantized weights
  (``unsloth/Llama-3.2-1B-Instruct``) in fp16 directly onto the GPU via
  PyTorch's MPS backend — plenty fast on an Apple Silicon Mac, no
  quantization needed given typical unified-memory headroom.
"""

import logging
import os

import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

DEFAULT_CUDA_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
DEFAULT_LOCAL_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

#: Override the model without touching code:
#:
#:     TRAVEL_ASSISTANT_MODEL=unsloth/Llama-3.2-3B-Instruct python -m travel_assistant.app
#:     TRAVEL_ASSISTANT_MODEL=unsloth/Llama-3.2-3B-Instruct python -m evals.extraction_eval
#:
#: The 1B default is what the proposal specifies and what the architecture
#: is designed around — routing stays in Python, the budget figure is
#: computed deterministically, and extraction is backstopped in code — so a
#: larger model is an upgrade, not a prerequisite. Point the eval harness at
#: a candidate to see what it actually buys before adopting it; the tradeoff
#: is download size, memory and latency per turn.
MODEL_ENV_VAR = "TRAVEL_ASSISTANT_MODEL"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_chat_model(
    model_id: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> ChatHuggingFace:
    """Load the local model and wrap it as a LangChain chat model.

    Picks CUDA, then Apple Silicon MPS, then CPU automatically. Resolution
    order for the checkpoint: the ``model_id`` argument, then the
    ``TRAVEL_ASSISTANT_MODEL`` environment variable, then the per-device
    default.
    """
    device = _pick_device()
    if model_id is None:
        model_id = os.environ.get(MODEL_ENV_VAR) or (
            DEFAULT_CUDA_MODEL_ID if device == "cuda" else DEFAULT_LOCAL_MODEL_ID
        )
    logger.info("loading %s on %s", model_id, device)

    pipeline_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "repetition_penalty": 1.1,
        "return_full_text": False,
    }

    if device == "cuda":
        # bitsandbytes needs accelerate's device_map dispatch for the
        # pre-quantized 4-bit weights.
        pipeline_llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            device_map="auto",
            pipeline_kwargs=pipeline_kwargs,
        )
    else:
        # langchain_huggingface's `from_model_id(device=...)` only accepts a
        # legacy CUDA device index, not device strings like "mps", so the
        # pipeline is built directly (per HuggingFacePipeline's own
        # documented "passing pipeline in directly" pattern) and handed off.
        dtype = torch.float16 if device == "mps" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            **pipeline_kwargs,
        )
        pipeline_llm = HuggingFacePipeline(pipeline=pipe)

    return ChatHuggingFace(llm=pipeline_llm)
