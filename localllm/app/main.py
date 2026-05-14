from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from app.api.routes import router

app = FastAPI(
    title="Local LLM Service",
    description="Internal local LLM service for the prompt-filter project",
    version="0.1.0",
)

REQUESTS_TOTAL = Counter(
    "localllm_requests_total",
    "Total requests handled by the local LLM service",
    ["endpoint"],
)

REQUEST_LATENCY = Histogram(
    "localllm_request_latency_seconds",
    "Request latency for local LLM service",
    ["endpoint"],
)

app.include_router(router)


@app.get("/metrics")
def metrics() -> Response:
    REQUESTS_TOTAL.labels(endpoint="/metrics").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
