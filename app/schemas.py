from typing import Any, Literal
from pydantic import BaseModel, Field


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    duration_seconds: float = Field(
        default=4.0,
        ge=1.0,
        le=60.0,
    )
    fps: int = Field(default=24, ge=8, le=60)
    resolution: Literal["480p", "720p"] = "480p"
    seed: int | None = None

    # Optional image-conditioning input. The application stores a host-side
    # model-ready PNG under reference_assets/. FastVideoProvider translates
    # that host path to the container-visible /inputs/... path.
    initial_image: str | None = None

    # Optional custom generation canvas. These are populated by
    # ImagePreprocessor so image-to-video generation keeps the uploaded
    # image's width/height proportion instead of forcing 16:9.
    width: int | None = Field(
        default=None,
        ge=128,
        le=4096,
    )
    height: int | None = Field(
        default=None,
        ge=128,
        le=4096,
    )


class VideoJobCreated(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class VideoJobStatus(BaseModel):
    job_id: str
    status: Literal[
        "queued",
        "preparing",
        "generating",
        "completed",
        "failed",
    ]
    progress: float = Field(ge=0.0, le=1.0)
    output_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )
