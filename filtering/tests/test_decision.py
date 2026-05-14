from app.api.schemas import Reason
from app.core.decision.decision_engine import DecisionEngine
from app.core.ml.inference import MLResult
from app.core.preprocess.normalize import ProcessedPrompt
from app.core.rules.rules_engine import RuleResult


def _processed() -> ProcessedPrompt:
    return ProcessedPrompt(raw_text="", normalized_text="", roles=[], metadata={}, features={})


def test_hard_block_decision():
    engine = DecisionEngine()

    decision = engine.decide(
        RuleResult(
            reasons=[Reason(source="rules", code="R001", message="x", severity="critical", action="block")],
            hard_block=True,
            score=0.9,
        ),
        MLResult(score=0.1, matched_features=[]),
        _processed(),
    )

    assert decision.decision == "block"
    assert decision.risk_score >= engine.block_threshold


def test_rewrite_decision_for_rewrite_action():
    engine = DecisionEngine()

    decision = engine.decide(
        RuleResult(
            reasons=[Reason(source="rules", code="R006", message="x", severity="medium", action="rewrite")],
            hard_block=False,
            score=0.55,
        ),
        MLResult(score=0.0, matched_features=[]),
        _processed(),
    )

    assert decision.decision == "rewrite"


def test_allow_decision_for_low_risk_prompt():
    engine = DecisionEngine()

    decision = engine.decide(
        RuleResult(reasons=[], hard_block=False, score=0.0),
        MLResult(score=0.0, matched_features=[]),
        _processed(),
    )

    assert decision.decision == "allow"
    assert decision.risk_score == 0.0
    assert decision.reasons == []


def test_ml_elevated_score_adds_reason_and_can_rewrite():
    engine = DecisionEngine(ml_threshold=0.25, rewrite_threshold=0.50, block_threshold=0.85)

    decision = engine.decide(
        RuleResult(reasons=[], hard_block=False, score=0.0),
        MLResult(score=0.6, matched_features=["priority_override"]),
        _processed(),
    )

    assert decision.decision == "rewrite"
    assert "ML001" in {reason.code for reason in decision.reasons}


def test_score_above_block_threshold_blocks_even_without_explicit_action():
    engine = DecisionEngine(block_threshold=0.85, rewrite_threshold=0.50, ml_threshold=0.95)

    decision = engine.decide(
        RuleResult(reasons=[], hard_block=False, score=1.0),
        MLResult(score=0.6, matched_features=[]),
        _processed(),
    )

    assert decision.decision == "block"
    assert decision.risk_score >= engine.block_threshold
