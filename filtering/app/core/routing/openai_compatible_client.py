import os
from typing import Any, Dict

import requests

from app.core.routing.response_adapter import (
    extract_openai_compatible_text,
    normalize_llm_response,
)


OPENAI_COMPATIBLE_BASE_URL = os.getenv("LLM_URL", "").rstrip("/")
OPENAI_COMPATIBLE_API_KEY = os.getenv("LLM_API_KEY", "")
OPENAI_COMPATIBLE_MODEL = os.getenv("LLM_MODEL_NAME", "")


def _get_required_env(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


def call_openai_compatible(
    prompt: str,
    timeout: int = 60,
) -> Dict[str, Any]:
    try:
        base_url = _get_required_env(
            "OPENAI_COMPATIBLE_BASE_URL",
            OPENAI_COMPATIBLE_BASE_URL,
        )
        api_key = _get_required_env(
            "OPENAI_COMPATIBLE_API_KEY",
            OPENAI_COMPATIBLE_API_KEY,
        )
        model = _get_required_env(
            "OPENAI_COMPATIBLE_MODEL",
            OPENAI_COMPATIBLE_MODEL,
        )

    except RuntimeError as exc:
        return normalize_llm_response(
            provider="openai_compatible",
            text="",
            raw={},
            error=str(exc),
            status_code=None,
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("LLM_MAX_NEW_TOKENS", "256")),
    }

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json()

    except requests.RequestException as exc:
        return normalize_llm_response(
            provider="openai_compatible",
            text="",
            raw={},
            error=str(exc),
            status_code=getattr(getattr(exc, "response", None), "status_code", None),
        )

    text = extract_openai_compatible_text(raw)

    return normalize_llm_response(
        provider="openai_compatible",
        text=text,
        raw=raw,
        status_code=response.status_code,
    )