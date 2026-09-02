from typing import Any, Literal
from pydantic import BaseModel, Field


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    duration_seconds: float = Field(default=4.0, ge=1.0, le=60.0)
    fps: int = Field(default=24, ge=8, le=60)
    resolution: Literal["480p", "720p"] = "480p"
    seed: int | None = None


class VideoJobCreated(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class VideoJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "preparing", "generating", "completed", "failed"]
    progress: float = Field(ge=0.0, le=1.0)
    output_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
