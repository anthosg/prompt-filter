from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["system", "developer", "user", "assistant", "tool"]
Decision = Literal["allow", "block", "rewrite"]

MLModelName = Literal[
    "tfidf_logreg",
    "multilingual_minilm",
    "tfidf_logreg_plus_multilingual_minilm",
    "transformers_distilbert",
    "none",
]


class Message(BaseModel):
    role: Role
    content: str = Field(..., min_length=1)


class DebugFilterOverrides(BaseModel):
    enable_rule_detection: Optional[bool] = None
    enable_heuristics: Optional[bool] = None
    enable_rewrite: Optional[bool] = None
    llm_enabled: Optional[bool] = None

    # Единое поле для управления ML-детектором.
    #
    # tfidf_logreg:
    #   текущий Sklearn TF-IDF + LogisticRegression
    #
    # multilingual_minilm:
    #   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    #   + LogisticRegression
    #
    # tfidf_logreg_plus_multilingual_minilm:
    #   ансамбль TF-IDF + MiniLM
    #
    # transformers_distilbert:
    #   transformer-классификатор, если он используется
    #
    # none:
    #   отключить ML-детекцию для конкретного запроса
    ml_model: Optional[MLModelName] = None


class FilterRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)
    context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    debug_filter: Optional[DebugFilterOverrides] = None


class Reason(BaseModel):
    source: str
    code: str
    message: str
    severity: str = "info"
    action: Optional[str] = None


class FilterResponse(BaseModel):
    decision: Decision
    risk_score: float = Field(..., ge=0.0, le=1.0)
    reasons: List[Reason]
    sanitized_request: Optional[FilterRequest] = None


class HealthResponse(BaseModel):
    status: str
    service: str


class ChatResponse(BaseModel):
    decision: Decision
    risk_score: float = Field(..., ge=0.0, le=1.0)
    reasons: List[Reason]
    sanitized_request: Optional[FilterRequest] = None
    ml_features: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    response: Optional[str] = None
    usage: Optional[dict] = None
    blocked: bool = False
    llm_called: bool = False
    llm_error: Optional[str] = None
    llm_status_code: Optional[int] = None
