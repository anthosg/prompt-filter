from typing import List

from app.api.schemas import Reason
from app.core.preprocess.normalize import ProcessedPrompt


def detect_obfuscation(processed: ProcessedPrompt) -> List[Reason]:
    reasons: List[Reason] = []
    features = processed.features

    if features.get("zero_width_count", 0) > 0:
        reasons.append(
            Reason(
                source="rules",
                code="H001",
                severity="medium",
                action="rewrite",
                message="Zero-width Unicode characters detected",
            )
        )

    if features.get("has_base64") or features.get("has_hex"):
        reasons.append(
            Reason(
                source="rules",
                code="H002",
                severity="medium",
                action="rewrite",
                message="Potential base64/hex obfuscation detected",
            )
        )

    return reasons
