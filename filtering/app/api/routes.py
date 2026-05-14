import os
import time
from dataclasses import dataclass

from fastapi import APIRouter, Depends

from app.api.dependencies import (
    get_decision_engine,
    get_ml_detector,
    get_rules_engine,
)
from app.api.schemas import (
    ChatResponse,
    FilterRequest,
    FilterResponse,
    HealthResponse,
)
from app.core.decision.decision_engine import DecisionEngine
from app.core.logging.logger import log_filter_event
from app.core.metrics.collectors import FILTER_DECISIONS, FILTER_LATENCY, FILTER_REQUESTS
from app.core.ml.inference import MLDetector
from app.core.preprocess.normalize import preprocess_request
from app.core.rewrite.sanitizer import rewrite_request
from app.core.routing.llm_router import route_to_llm
from app.core.rules.rules_engine import RulesEngine


router = APIRouter()


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FilterFlags:
    enable_rule_detection: bool
    enable_heuristics: bool
    enable_rewrite: bool
    llm_enabled: bool
    ml_model: str


def _override_bool(current: bool, override):
    if override is None:
        return current

    return bool(override)


def _normalize_ml_model(value: str | None) -> str:
    if value is None:
        return "tfidf_logreg"

    normalized = str(value).strip().lower()

    legacy_aliases = {
        "sklearn": "tfidf_logreg",
        "tfidf": "tfidf_logreg",
        "logreg": "tfidf_logreg",
        "embedding": "multilingual_minilm",
        "minilm": "multilingual_minilm",
        "ensemble": "tfidf_logreg_plus_multilingual_minilm",
        "transformers": "transformers_distilbert",
        "transformer": "transformers_distilbert",
        "off": "none",
        "false": "none",
        "disabled": "none",
    }

    return legacy_aliases.get(normalized, normalized)


def resolve_filter_flags(request: FilterRequest) -> FilterFlags:
    flags = FilterFlags(
        enable_rule_detection=_env_flag("ENABLE_RULE_DETECTION", default=True),
        enable_heuristics=_env_flag("ENABLE_HEURISTICS", default=True),
        enable_rewrite=_env_flag("ENABLE_REWRITE", default=True),
        llm_enabled=_env_flag("LLM_ENABLED", default=True),

        # По умолчанию — sklearn TF-IDF + LogisticRegression.
        ml_model=_normalize_ml_model(os.getenv("ML_MODEL", "tfidf_logreg")),
    )

    if not _env_flag("ENABLE_FILTER_OVERRIDES", default=False):
        return flags

    debug = getattr(request, "debug_filter", None)

    if debug is None:
        return flags

    return FilterFlags(
        enable_rule_detection=_override_bool(
            flags.enable_rule_detection,
            debug.enable_rule_detection,
        ),
        enable_heuristics=_override_bool(
            flags.enable_heuristics,
            debug.enable_heuristics,
        ),
        enable_rewrite=_override_bool(
            flags.enable_rewrite,
            debug.enable_rewrite,
        ),
        llm_enabled=_override_bool(
            flags.llm_enabled,
            debug.llm_enabled,
        ),
        ml_model=(
            _normalize_ml_model(debug.ml_model)
            if debug.ml_model is not None
            else flags.ml_model
        ),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="filtering")


@router.post("/v1/chat", response_model=ChatResponse)
def chat(
    request: FilterRequest,
    rules_engine: RulesEngine = Depends(get_rules_engine),
    ml_detector: MLDetector = Depends(get_ml_detector),
    decision_engine: DecisionEngine = Depends(get_decision_engine),
) -> ChatResponse:
    start = time.perf_counter()

    FILTER_REQUESTS.labels(endpoint="/v1/chat").inc()

    processed = preprocess_request(request)
    flags = resolve_filter_flags(request)

    if flags.enable_rule_detection or flags.enable_heuristics:
        rule_result = rules_engine.analyze(
            processed,
            enable_rules=flags.enable_rule_detection,
            enable_heuristics=flags.enable_heuristics,
        )
    else:
        rule_result = rules_engine.empty_result()

    # ML включается и выключается только через ml_model.
    #
    # Примеры:
    # ml_model = "tfidf_logreg"
    # ml_model = "multilingual_minilm"
    # ml_model = "tfidf_logreg_plus_multilingual_minilm"
    # ml_model = "transformers_distilbert"
    # ml_model = "none"
    ml_result = ml_detector.analyze(
        processed,
        model_override=flags.ml_model,
    )

    decision_result = decision_engine.decide(
        rule_result,
        ml_result,
        processed,
    )

    sanitized = None

    if decision_result.decision == "rewrite":
        if flags.enable_rewrite:
            sanitized = rewrite_request(request, decision_result.reasons)
        else:
            decision_result.decision = "block"

    filter_response = FilterResponse(
        decision=decision_result.decision,
        risk_score=decision_result.risk_score,
        reasons=decision_result.reasons,
        sanitized_request=sanitized,
    )

    FILTER_DECISIONS.labels(decision=decision_result.decision).inc()

    if decision_result.decision == "block":
        latency = time.perf_counter() - start
        FILTER_LATENCY.labels(endpoint="/v1/chat").observe(latency)

        log_filter_event(
            request,
            filter_response,
            latency_ms=round(latency * 1000, 2),
        )

        return ChatResponse(
            decision=decision_result.decision,
            risk_score=decision_result.risk_score,
            reasons=decision_result.reasons,
            sanitized_request=None,
            ml_features=ml_result.matched_features,
            blocked=True,
            llm_called=False,
            response="Request was blocked by prompt-filter.",
        )

    if sanitized is not None:
        prompt = _extract_prompt_from_request(sanitized)
    else:
        prompt = processed.normalized_text

    if not flags.llm_enabled:
        latency = time.perf_counter() - start
        FILTER_LATENCY.labels(endpoint="/v1/chat").observe(latency)

        log_filter_event(
            request,
            filter_response,
            latency_ms=round(latency * 1000, 2),
        )

        return ChatResponse(
            decision=decision_result.decision,
            risk_score=decision_result.risk_score,
            reasons=decision_result.reasons,
            sanitized_request=sanitized,
            ml_features=ml_result.matched_features,
            blocked=False,
            llm_called=False,
            model=None,
            response=None,
            usage=None,
        )

    llm_response = route_to_llm(prompt)

    latency = time.perf_counter() - start
    FILTER_LATENCY.labels(endpoint="/v1/chat").observe(latency)

    log_filter_event(
        request,
        filter_response,
        latency_ms=round(latency * 1000, 2),
    )

    return ChatResponse(
        decision=decision_result.decision,
        risk_score=decision_result.risk_score,
        reasons=decision_result.reasons,
        sanitized_request=sanitized,
        ml_features=ml_result.matched_features,
        blocked=False,
        llm_called=True,
        model=llm_response.get("model"),
        response=llm_response.get("response"),
        usage=llm_response.get("usage"),
        llm_error=llm_response.get("error"),
        llm_status_code=llm_response.get("status_code"),
    )


def _extract_prompt_from_request(request: FilterRequest) -> str:
    parts = []

    for message in request.messages:
        if message.role == "user":
            parts.append(message.content)

    if request.context:
        parts.append(request.context)

    return "\n".join(parts)