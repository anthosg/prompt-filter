import os
from typing import Any, Dict
import requests
from app.core.routing.response_adapter import normalize_llm_response


LLM_URL = os.getenv("LLM_URL", "http://localllm:8001")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_generation_payload(
    prompt: str,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "max_new_tokens": (
            max_new_tokens
            if max_new_tokens is not None
            else int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
        ),
        "temperature": (
            temperature
            if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "0.0"))
        ),
        "top_p": (
            top_p
            if top_p is not None
            else float(os.getenv("LLM_TOP_P", "1.0"))
        ),
        "do_sample": (
            do_sample
            if do_sample is not None
            else _env_bool("LLM_DO_SAMPLE", False)
        ),
    }


def call_localllm(
    prompt: str,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
    timeout: int = 60,
) -> Dict[str, Any]:

    payload = _get_generation_payload(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )

    try:
        response = requests.post(
            f"{LLM_URL}/v1/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json()

    except requests.RequestException as exc:
        return normalize_llm_response(
            provider="localllm",
            text="",
            raw={},
            error=str(exc),
            status_code=getattr(getattr(exc, "response", None), "status_code", None),
        )

    text = (
        raw.get("text")
        or raw.get("generated_text")
        or raw.get("response")
        or raw.get("answer")
        or ""
    )

    return normalize_llm_response(
        provider="localllm",
        text=text,
        raw=raw,
        status_code=response.status_code,
    )