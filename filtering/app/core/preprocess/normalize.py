import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List

from app.api.schemas import FilterRequest
from app.core.preprocess.encoding_detector import detect_encoding_signals
from app.core.preprocess.tokenizer import simple_tokenize
from app.core.preprocess.unicode_filter import count_zero_width, remove_zero_width


@dataclass
class ProcessedPrompt:
    raw_text: str
    normalized_text: str
    roles: List[str]
    metadata: Dict
    features: Dict


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = remove_zero_width(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_request(request: FilterRequest) -> ProcessedPrompt:
    raw_parts = []
    roles = []

    for message in request.messages:
        roles.append(message.role)
        raw_parts.append(f"[{message.role}] {message.content}")

    if request.context:
        raw_parts.append(f"[context] {request.context}")

    raw_text = "\n".join(raw_parts)
    normalized_text = normalize_text(raw_text)
    encoding_signals = detect_encoding_signals(raw_text)
    tokens = simple_tokenize(normalized_text)

    features = {
        "raw_len": len(raw_text),
        "normalized_len": len(normalized_text),
        "token_count": len(tokens),
        "zero_width_count": count_zero_width(raw_text),
        "has_base64": encoding_signals.has_base64,
        "has_hex": encoding_signals.has_hex,
        "base64_candidates": encoding_signals.base64_candidates,
        "hex_candidates": encoding_signals.hex_candidates,
        "has_context": request.context is not None,
    }

    return ProcessedPrompt(
        raw_text=raw_text,
        normalized_text=normalized_text,
        roles=roles,
        metadata=request.metadata,
        features=features,
    )
