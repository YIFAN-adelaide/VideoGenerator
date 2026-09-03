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

    async def health(self) -> dict[str, Any] | None:
        """Optional runtime health information for externally served providers."""
        return None

    async def close(self) -> None:
        """Release provider-owned lightweight resources such as HTTP clients."""
        return None
