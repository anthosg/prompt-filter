import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.api.schemas import FilterRequest, FilterResponse
from app.core.logging.json_formatter import to_json_line

LOG_DIR = Path(os.getenv("FILTERING_LOG_DIR", "/app/logs"))
LOG_FILE = LOG_DIR / "filtering.jsonl"


def log_filter_event(request: FilterRequest, response: FilterResponse, latency_ms: float) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4()),
        "decision": response.decision,
        "risk_score": response.risk_score,
        "reasons": [reason.model_dump() for reason in response.reasons],
        "message_count": len(request.messages),
        "has_context": request.context is not None,
        "latency_ms": latency_ms,
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(to_json_line(event) + "\n")