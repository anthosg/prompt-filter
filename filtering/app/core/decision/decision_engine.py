import os
import re
from dataclasses import dataclass
from typing import List

from app.api.schemas import Decision, Reason
from app.core.decision.risk_score import combine_scores
from app.core.ml.inference import MLResult
from app.core.preprocess.normalize import ProcessedPrompt
from app.core.rules.rules_engine import RuleResult

# Максимально быстрая компиляция паттернов
SECURITY_TERMS_PATTERN = re.compile(
    r"(" + r"|".join([
        r"prompt[- ]injection", r"jailbreak", r"system prompt", r"developer message",
        r"llm security", r"ai security", r"guardrail", r"red[- ]team(?:ing)?", 
        r"injection attack", r"инъекция", r"prompt-инъекция", r"джейлбрейк", 
        r"системный промпт", r"безопасность llm"
    ]) + r")", 
    re.IGNORECASE
)

SAFE_DISCUSSION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bwhat is\b", r"\bexplain\b", r"\bdescribe\b", r"\bcompare\b",
        r"\bdifference between\b", r"\bhow to detect\b", r"\bhow to prevent\b",
        r"\bmitigation\b", r"\bdefense\b", r"\bprotection\b", r"\bобъясни\b",
        r"\bопиши\b", r"\bчто такое\b", r"\bкак обнаруж", r"\bкак предотврат",
        r"\bметоды защиты\b", r"\bчем отличается\b", r"\bбезопасн", r"\bзащит",
    ]
]

DANGEROUS_INTENT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(?<!how to )(?<!как )\bignore\s+(all\s+)?previous",
        r"(?<!how to )(?<!как )\bignore\s+.*instructions?",
        r"(?<!how to )(?<!как )\bdisregard\s+.*instructions?",
        r"(?<!how to )(?<!как )\bforget\s+(all\s+)?previous",
        r"(?<!how to )(?<!как )\boverride\s+.*instructions?",
        r"(?<!how to )(?<!как )\bbypass\s+.*rules?",
        r"(?<!how to )(?<!как )\bbypass\s+.*restrictions?",
        r"\breveal\s+.*system\s+prompt",
        r"\bshow\s+.*system\s+prompt",
        r"\bprint\s+.*system\s+prompt",
        r"\bact\s+as\s+(dan|developer|system)",
        r"\bdecode\s+.*and\s+execute",
        r"\bexecute\s+.*hidden",
        r"\bfollow\s+.*hidden\s+instructions?",
        r"\bleak\s+.*prompt",
        r"(?<!как )\bигнорируй\s+.*инструкц",
        r"(?<!как )\bобойди\s+.*огранич",
        r"\bраскрой\s+.*системн",
        r"\bпокажи\s+.*системн",
        r"\bвыведи\s+.*системн",
        r"\bвыполни\s+.*скрыт",
    ]
]

def _contains_any_regex(text: str, patterns: list[re.Pattern]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def is_borderline_safe_discussion(text: str) -> bool:
    if not text:
        return False
    
    if not SECURITY_TERMS_PATTERN.search(text):
        return False

    has_safe_intent = _contains_any_regex(text, SAFE_DISCUSSION_PATTERNS)
    
    # Для обеспечения Hard Override мы игнорируем опасные цитаты, 
    # если найден явный исследовательский интент (how to prevent и т.д.)
    # Это решает проблему "The Quote Problem".
    if has_safe_intent:
        return True 

    has_dangerous_intent = _contains_any_regex(text, DANGEROUS_INTENT_PATTERNS)
    return has_safe_intent and not has_dangerous_intent


@dataclass
class DecisionResult:
    decision: Decision
    risk_score: float
    reasons: List[Reason]


class DecisionEngine:
    def __init__(
        self,
        block_threshold: float = 0.88,
        rewrite_threshold: float = 0.62,
        ml_threshold: float = 0.60,
        safe_discussion_max_risk: float = 0.18,
    ):
        self.block_threshold = block_threshold
        self.rewrite_threshold = rewrite_threshold
        self.ml_threshold = ml_threshold
        self.safe_discussion_max_risk = safe_discussion_max_risk

    @classmethod
    def from_default_config(cls) -> "DecisionEngine":
        return cls(
            block_threshold=float(os.getenv("RISK_SCORE_BLOCK", "0.88")),
            rewrite_threshold=float(os.getenv("RISK_SCORE_REWRITE", "0.62")),
            ml_threshold=float(os.getenv("ML_THRESHOLD", "0.60")), 
            safe_discussion_max_risk=float(os.getenv("SAFE_DISCUSSION_MAX_RISK", "0.18")),
        )

    def decide(
        self,
        rule_result: RuleResult,
        ml_result: MLResult,
        processed: ProcessedPrompt,
    ) -> DecisionResult:
        reasons = list(rule_result.reasons)

        if ml_result.score >= self.ml_threshold:
            reasons.append(
                Reason(
                    source="ml",
                    code="ML001",
                    severity="medium",
                    action="rewrite",
                    message=f"ML detector assigned elevated risk ({ml_result.score:.2f})",
                )
            )

        has_block_action = any(reason.action == "block" for reason in reasons)
        has_rewrite_action = any(reason.action == "rewrite" for reason in reasons)
        has_critical = any(reason.severity == "critical" for reason in reasons)

        risk_score = combine_scores(rule_result.score, ml_result.score)

        prompt_text = processed.normalized_text or processed.raw_text or ""
        safe_discussion = is_borderline_safe_discussion(prompt_text)

        # 1. Приоритет безопасного обсуждения (Hard Override)
        is_suspiciously_long = len(prompt_text) > 1500  

        if safe_discussion and not rule_result.hard_block and not has_critical:
            if not is_suspiciously_long:
                return DecisionResult(
                    decision="allow",
                    risk_score=min(risk_score, self.safe_discussion_max_risk),
                    reasons=[
                        Reason(
                            source="heuristics",
                            code="SAFE_DISCUSSION",
                            severity="low",
                            action="allow",
                            message="Legitimate LLM security discussion (ML override applied)",
                        )
                    ],
                )

        # 2. Жесткие блокировки (Hard Blocks)
        if rule_result.hard_block or has_critical or has_block_action:
            risk_score = max(risk_score, self.block_threshold)
            return DecisionResult(
                decision="block",
                risk_score=min(risk_score, 1.0),
                reasons=reasons,
            )

        # 3. Базовая маршрутизация на основе вычисленного risk_score
        if risk_score >= self.block_threshold:
            decision: Decision = "block"
        elif risk_score >= self.rewrite_threshold or has_rewrite_action:
            decision = "rewrite"
            risk_score = max(risk_score, self.rewrite_threshold) 
        else:
            decision = "allow"

        return DecisionResult(
            decision=decision,
            risk_score=min(risk_score, 1.0),
            reasons=reasons,
        )