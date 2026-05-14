import json
from datetime import datetime, timezone
from typing import Any, Dict


def to_json_line(payload: Dict[str, Any]) -> str:
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    return json.dumps(payload, ensure_ascii=False)
