from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_per_sec: float
