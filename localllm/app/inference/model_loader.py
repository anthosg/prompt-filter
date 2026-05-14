import os
from functools import lru_cache
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.api.schemas import ModelInfoResponse


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def get_model_name() -> str:
    return os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL_NAME)


def get_model_path() -> str:
    model_path = os.getenv("LLM_MODEL_PATH", "").strip()
    return model_path if model_path else get_model_name()


def get_device() -> str:
    configured_device = os.getenv("LLM_DEVICE", "auto").strip().lower()

    if configured_device in {"cpu", "cuda"}:
        if configured_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return configured_device

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_dtype(device: str):
    if device == "cuda":
        return torch.float16
    return torch.float32


@lru_cache(maxsize=1)
def load_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    model_path = get_model_path()
    device = get_device()
    torch_dtype = get_torch_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def get_model_info() -> ModelInfoResponse:
    model_name = get_model_name()
    model_path = get_model_path()
    device = get_device()

    try:
        load_model()
        loaded = True
    except Exception:
        loaded = False

    return ModelInfoResponse(
        model_name=model_name,
        model_path=model_path,
        device=device,
        loaded=loaded,
    )
