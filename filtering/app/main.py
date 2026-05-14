from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.routes import router

app = FastAPI(
    title="Prompt Filter - Filtering Service",
    description="Prompt injection filtering service for LLM API requests",
    version="0.1.0",
)

app.include_router(router)


@app.get("/metrics")
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )