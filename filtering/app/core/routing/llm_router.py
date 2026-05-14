import os
from typing import Any, Callable, Dict

from app.core.routing.localllm_client import call_localllm
from app.core.routing.openai_compatible_client import call_openai_compatible


LLMProvider = Callable[[str], Dict[str, Any]]


PROVIDERS: dict[str, LLMProvider] = {
    "localllm": call_localllm,
    "openai_compatible": call_openai_compatible,
}


def get_default_provider() -> str:
    return os.getenv("LLM_PROVIDER", "localllm")


def get_supported_providers() -> list[str]:
    return sorted(PROVIDERS.keys())


def route_to_llm(
    prompt: str,
    provider: str | None = None,
) -> Dict[str, Any]:
    selected_provider = provider or get_default_provider()

    llm_client = PROVIDERS.get(selected_provider)
    if llm_client is None:
        supported = ", ".join(get_supported_providers())
        raise ValueError(
            f"Unsupported LLM provider: {selected_provider}. "
            f"Supported providers: {supported}"
        )

    return llm_client(prompt)
