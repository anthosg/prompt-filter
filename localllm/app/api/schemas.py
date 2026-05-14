from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt for local LLM")
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    do_sample: bool = False


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class GenerateResponse(BaseModel):
    model: str
    response: str
    usage: TokenUsage


class ModelInfoResponse(BaseModel):
    model_name: str
    model_path: str
    device: str
    loaded: bool


class HealthResponse(BaseModel):
    status: str
    service: str
