from typing import Any, Literal, TypedDict


JobState = Literal[
    "queued",
    "preparing",
    "generating",
    "completed",
    "failed",
]


class VideoState(TypedDict, total=False):
    job_id: str
    request: dict[str, Any]
    status: JobState
    progress: float
    output_path: str | None
    error: str | None
    metadata: dict[str, Any]
