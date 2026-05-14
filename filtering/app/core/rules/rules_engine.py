import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from app.api.schemas import Reason
from app.core.preprocess.normalize import ProcessedPrompt
from app.core.rules.heuristics import detect_obfuscation
from app.core.rules.policies import detect_context_policies


@dataclass
class RuleResult:
    reasons: List[Reason]
    hard_block: bool
    score: float


class RulesEngine:
    def __init__(self, ruleset_path: str = "app/configs/ruleset.yaml"):
        self.ruleset_path = Path(ruleset_path)
        self.rules = self._load_rules()

    @classmethod
    def from_default_config(cls) -> "RulesEngine":
        return cls("app/configs/ruleset.yaml")

    def _load_rules(self) -> List[Dict[str, Any]]:
        if not self.ruleset_path.exists():
            raise FileNotFoundError(f"Ruleset not found: {self.ruleset_path}")

        with self.ruleset_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        rules = data.get("rules", [])

        if not isinstance(rules, list):
            raise ValueError("Invalid ruleset.yaml: 'rules' must be a list")

        return rules

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _is_hard_block(action: str, severity: str) -> bool:
        action = (action or "").lower()
        severity = (severity or "").lower()

        return action in {"block", "hard"} or severity == "critical"

    def analyze(
        self,
        processed: ProcessedPrompt,
        enable_rules: bool = True,
        enable_heuristics: bool = True,
    ) -> RuleResult:
        reasons: List[Reason] = []
        hard_block = False
        score = 0.0

        normalized_text = self._normalize_for_match(processed.normalized_text)

        # Signature-based rules
        if enable_rules:
            for rule in self.rules:
                patterns = rule.get("patterns", [])
                rule_score = float(rule.get("score", 0.0))
                severity = str(rule.get("severity", "medium"))
                action = str(rule.get("action", "rewrite"))

                for pattern in patterns:
                    normalized_pattern = self._normalize_for_match(str(pattern))

                    if normalized_pattern and normalized_pattern in normalized_text:
                        reasons.append(
                            Reason(
                                source="rules",
                                code=str(rule.get("code", "UNKNOWN")),
                                severity=severity,
                                action=action,
                                message=str(
                                    rule.get(
                                        "description",
                                        rule.get("name", "Rule matched"),
                                    )
                                ),
                            )
                        )

                        if self._is_hard_block(action, severity):
                            hard_block = True

                        score = max(score, rule_score)
                        break

        heuristic_score = float(os.getenv("HEURISTIC_SCORE", "0.55"))
        policy_score = float(os.getenv("POLICY_SCORE", "0.75"))

        # Heuristics
        if enable_heuristics:
            heuristic_reasons = detect_obfuscation(processed)
            policy_reasons = detect_context_policies(processed)

            reasons.extend(heuristic_reasons)
            reasons.extend(policy_reasons)

            if heuristic_reasons:
                score = max(score, heuristic_score)

            if policy_reasons:
                score = max(score, policy_score)

        score = min(score, 1.0)

        return RuleResult(
            reasons=reasons,
            hard_block=hard_block,
            score=score,
        )

    def empty_result(self) -> RuleResult:
        return RuleResult(
            reasons=[],
            hard_block=False,
            score=0.0,
        )