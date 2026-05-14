from typing import List

from app.api.schemas import Reason
from app.core.preprocess.normalize import ProcessedPrompt


INDIRECT_INJECTION_MARKERS = [
    "the document says",
    "document says",
    "в документе написано",
    "следуй инструкциям из документа",
]


def detect_context_policies(processed: ProcessedPrompt) -> List[Reason]:
    text = processed.normalized_text.lower()
    reasons: List[Reason] = []

    if processed.features.get("has_context"):
        for marker in INDIRECT_INJECTION_MARKERS:
            if marker in text:
                reasons.append(
                    Reason(
                        source="rules",
                        code="P001",
                        severity="medium",
                        message="Potential indirect prompt injection in context",
                    )
                )
                break

    return reasons
