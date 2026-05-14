from app.core.decision.decision_engine import DecisionEngine
from app.core.ml.inference import MLDetector
from app.core.rules.rules_engine import RulesEngine

_rules_engine = RulesEngine.from_default_config()
_ml_detector = MLDetector()
_decision_engine = DecisionEngine.from_default_config()


def get_rules_engine() -> RulesEngine:
    return _rules_engine


def get_ml_detector() -> MLDetector:
    return _ml_detector


def get_decision_engine() -> DecisionEngine:
    return _decision_engine
