from typing import Any, Dict


def normalize_llm_response(
    *,
    provider: str,
    text: str,
    raw: Dict[str, Any] | None = None,
    error: str | None = None,
    status_code: int | None = None,
) -> Dict[str, Any]:
    raw = raw or {}

    response_text = (
        text
        or raw.get("response")
        or raw.get("text")
        or raw.get("generated_text")
        or raw.get("answer")
        or None
    )

    return {
        "provider": provider,
        "model": raw.get("model"),
        "response": response_text,
        "usage": raw.get("usage"),
        "text": response_text or "",
        "raw": raw,
        "error": error,
        "status_code": status_code,
    }


def extract_openai_compatible_text(raw: Dict[str, Any]) -> str:
    try:
        return raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
