import asyncio
import json
from pathlib import Path

from app.providers.base import ProviderResult, VideoProvider
from app.schemas import VideoGenerationRequest


class MockVideoProvider(VideoProvider):
    """Cheap local provider used to validate orchestration without a GPU."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate(
        self,
        request: VideoGenerationRequest,
        job_id: str,
    ) -> ProviderResult:
        await asyncio.sleep(0.25)

        output_path = self.output_dir / f"{job_id}.mock.json"
        payload = {
            "job_id": job_id,
            "provider": "mock",
            "request": request.model_dump(),
            "note": "Replace MockVideoProvider with HeliosProvider on the AWS GPU host.",
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return ProviderResult(
            output_path=str(output_path.resolve()),
            metadata={"provider": "mock"},
        )
