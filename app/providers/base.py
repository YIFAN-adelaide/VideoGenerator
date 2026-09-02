from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.schemas import VideoGenerationRequest


@dataclass
class ProviderResult:
    output_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        request: VideoGenerationRequest,
        job_id: str,
    ) -> ProviderResult:
        """Generate one video and return its output path and metadata."""
        raise NotImplementedError
