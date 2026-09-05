from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from app.director.video_plan import ShotPlan
from app.services.fastvideo_shot_generator import (
    FastVideoShotGenerator,
)
from app.services.image_probe import ImageProbeResult
from app.services.video_probe import VideoProbeResult


class FakeVideoProvider:
    def __init__(
        self,
        output_path: Path,
    ) -> None:
        self.output_path = output_path
        self.requests = []

    async def generate(
        self,
        request,
        job_id: str,
    ):
        self.requests.append(request)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.output_path.write_bytes(
            b"fake-video"
        )

        return SimpleNamespace(
            output_path=str(self.output_path),
            metadata={},
        )


class FakeImageProbe:
    async def probe(
        self,
        path: str | Path,
    ) -> ImageProbeResult:
        resolved = Path(path).resolve()

        return ImageProbeResult(
            path=resolved,
            width=1280,
            height=704,
            aspect_ratio=1280 / 704,
            pixel_format="rgb24",
        )


class FakeVideoProbe:
    async def probe(
        self,
        path: str | Path,
    ) -> VideoProbeResult:
        return VideoProbeResult(
            path=Path(path).resolve(),
            duration_seconds=5.041667,
            fps=24.0,
            frame_count=121,
            width=1280,
            height=704,
        )


def _shot() -> ShotPlan:
    return ShotPlan(
        shot_id="shot_002",
        duration_seconds=5,
        fps=24,
        resolution="720p",
        action="The tiger keeps walking.",
        location="Snowy forest",
        characters=["tiger"],
        camera="tracking shot",
        lighting="winter daylight",
        generation_prompt_en=(
            "The tiger continues walking through the snow."
        ),
        generation_prompt_zh="老虎继续走过雪地。",
        prompt="The tiger continues walking through the snow.",
        metadata={},
    )


def test_fastvideo_shot_generator_passes_previous_frame_as_initial_image(
    tmp_path: Path,
) -> None:
    provider = FakeVideoProvider(
        tmp_path / "provider.mp4"
    )

    previous_frame = (
        tmp_path
        / "reference_assets"
        / "jobs"
        / "job-001"
        / "shot_001_last.png"
    )
    previous_frame.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    previous_frame.write_bytes(
        b"fake-png"
    )

    generator = FastVideoShotGenerator(
        provider,
        video_probe=FakeVideoProbe(),
        image_probe=FakeImageProbe(),
    )

    target = (
        tmp_path
        / "job-001"
        / "shots"
        / "shot_002.mp4"
    )

    asyncio.run(
        generator.generate_shot(
            shot=_shot(),
            prompt="The tiger continues walking.",
            output_path=target,
            initial_image=previous_frame,
        )
    )

    request = provider.requests[0]

    assert request.initial_image == str(
        previous_frame.resolve()
    )
    assert request.width == 1280
    assert request.height == 704
    assert request.duration_seconds == 5
    assert request.fps == 24
