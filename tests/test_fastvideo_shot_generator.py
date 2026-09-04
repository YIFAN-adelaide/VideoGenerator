from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.director.video_plan import ShotPlan
from app.services.fastvideo_shot_generator import (
    FastVideoShotGenerator,
    FastVideoShotGeneratorError,
)
from app.services.generated_shot import GeneratedShot
from app.services.video_probe import VideoProbeResult


class FakeVideoProvider:
    def __init__(
        self,
        provider_output: Path,
    ) -> None:
        self.provider_output = provider_output
        self.requests = []
        self.job_ids = []

    async def generate(
        self,
        request,
        job_id: str,
    ):
        self.requests.append(request)
        self.job_ids.append(job_id)

        self.provider_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.provider_output.write_bytes(
            b"fake-provider-video"
        )

        return SimpleNamespace(
            output_path=str(
                self.provider_output
            ),
            metadata={
                "provider": "fastvideo",
            },
        )


class FakeVideoProbe:
    def __init__(
        self,
        *,
        duration_seconds: float = 4.875,
        fps: float = 24.0,
        frame_count: int = 117,
        width: int = 1280,
        height: int = 704,
    ) -> None:
        self.duration_seconds = duration_seconds
        self.fps = fps
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.paths: list[Path] = []

    async def probe(
        self,
        path: str | Path,
    ) -> VideoProbeResult:
        resolved = Path(path).resolve()
        self.paths.append(resolved)

        return VideoProbeResult(
            path=resolved,
            duration_seconds=self.duration_seconds,
            fps=self.fps,
            frame_count=self.frame_count,
            width=self.width,
            height=self.height,
        )


def _shot(
    *,
    shot_id: str = "shot_001",
    duration_seconds: float = 5.0,
    fps: int = 24,
    resolution: str = "720p",
    metadata: dict | None = None,
) -> ShotPlan:
    return ShotPlan(
        shot_id=shot_id,
        duration_seconds=duration_seconds,
        fps=fps,
        resolution=resolution,
        action="A tiger walks forward.",
        location="Snowy forest",
        characters=["tiger"],
        camera="tracking shot",
        lighting="soft winter daylight",
        generation_prompt_en=(
            "A cinematic tiger walking through "
            "a snowy forest."
        ),
        generation_prompt_zh=(
            "一只老虎走过雪地森林的电影感镜头。"
        ),
        prompt=(
            "A cinematic tiger walking through "
            "a snowy forest."
        ),
        metadata=metadata or {},
    )


def test_adapter_maps_shot_to_video_generation_request(
    tmp_path: Path,
):
    provider_output = (
        tmp_path
        / "provider"
        / "generated.mp4"
    )

    provider = FakeVideoProvider(
        provider_output
    )
    probe = FakeVideoProbe()

    adapter = FastVideoShotGenerator(
        provider,
        video_probe=probe,
    )

    target = (
        tmp_path
        / "long-job-001"
        / "shots"
        / "shot_001.mp4"
    )

    result = asyncio.run(
        adapter.generate_shot(
            shot=_shot(),
            prompt=(
                "  A cinematic tiger in snow.  "
            ),
            output_path=target,
        )
    )

    assert isinstance(result, GeneratedShot)
    assert result.path == target.resolve()
    assert target.exists()

    assert result.requested_duration_seconds == 5.0
    assert result.actual_duration_seconds == pytest.approx(
        4.875
    )
    assert result.duration_delta_seconds == pytest.approx(
        -0.125
    )
    assert result.fps == pytest.approx(24.0)
    assert result.frame_count == 117
    assert result.width == 1280
    assert result.height == 704

    assert probe.paths == [
        target.resolve()
    ]

    assert len(provider.requests) == 1

    request = provider.requests[0]

    assert request.prompt == (
        "A cinematic tiger in snow."
    )
    assert request.duration_seconds == 5.0
    assert request.fps == 24
    assert request.resolution == "720p"
    assert request.seed is None

    assert provider.job_ids == [
        "long-job-001_shot_001"
    ]


def test_adapter_passes_seed_from_shot_metadata(
    tmp_path: Path,
):
    provider = FakeVideoProvider(
        tmp_path
        / "provider.mp4"
    )

    adapter = FastVideoShotGenerator(
        provider,
        video_probe=FakeVideoProbe(),
    )

    target = (
        tmp_path
        / "job"
        / "shots"
        / "shot_002.mp4"
    )

    asyncio.run(
        adapter.generate_shot(
            shot=_shot(
                shot_id="shot_002",
                metadata={
                    "seed": 123,
                },
            ),
            prompt="Tiger scene",
            output_path=target,
        )
    )

    assert provider.requests[0].seed == 123


def test_adapter_can_move_instead_of_copy(
    tmp_path: Path,
):
    provider_output = (
        tmp_path
        / "provider.mp4"
    )

    provider = FakeVideoProvider(
        provider_output
    )

    adapter = FastVideoShotGenerator(
        provider,
        preserve_provider_output=False,
        video_probe=FakeVideoProbe(),
    )

    target = (
        tmp_path
        / "job"
        / "shots"
        / "shot_001.mp4"
    )

    result = asyncio.run(
        adapter.generate_shot(
            shot=_shot(),
            prompt="Tiger scene",
            output_path=target,
        )
    )

    assert result.path.exists()
    assert not provider_output.exists()


def test_adapter_rejects_empty_prompt(
    tmp_path: Path,
):
    provider = FakeVideoProvider(
        tmp_path
        / "provider.mp4"
    )

    adapter = FastVideoShotGenerator(
        provider,
        video_probe=FakeVideoProbe(),
    )

    with pytest.raises(
        FastVideoShotGeneratorError,
        match="empty generation prompt",
    ):
        asyncio.run(
            adapter.generate_shot(
                shot=_shot(),
                prompt="   ",
                output_path=(
                    tmp_path
                    / "job"
                    / "shots"
                    / "shot_001.mp4"
                ),
            )
        )


def test_adapter_rejects_missing_provider_output(
    tmp_path: Path,
):
    class MissingOutputProvider:
        async def generate(
            self,
            request,
            job_id,
        ):
            return SimpleNamespace(
                output_path=str(
                    tmp_path
                    / "missing.mp4"
                ),
                metadata={},
            )

    adapter = FastVideoShotGenerator(
        MissingOutputProvider(),
        video_probe=FakeVideoProbe(),
    )

    with pytest.raises(
        FastVideoShotGeneratorError,
        match="does not exist",
    ):
        asyncio.run(
            adapter.generate_shot(
                shot=_shot(),
                prompt="Tiger scene",
                output_path=(
                    tmp_path
                    / "job"
                    / "shots"
                    / "shot_001.mp4"
                ),
            )
        )


def test_adapter_rejects_invalid_seed(
    tmp_path: Path,
):
    provider = FakeVideoProvider(
        tmp_path
        / "provider.mp4"
    )

    adapter = FastVideoShotGenerator(
        provider,
        video_probe=FakeVideoProbe(),
    )

    with pytest.raises(
        FastVideoShotGeneratorError,
        match="seed must be an integer",
    ):
        asyncio.run(
            adapter.generate_shot(
                shot=_shot(
                    metadata={
                        "seed": "abc",
                    },
                ),
                prompt="Tiger scene",
                output_path=(
                    tmp_path
                    / "job"
                    / "shots"
                    / "shot_001.mp4"
                ),
            )
        )


def test_adapter_wraps_probe_failure(
    tmp_path: Path,
):
    class FailingProbe:
        async def probe(self, path):
            raise RuntimeError("probe failed")

    provider = FakeVideoProvider(
        tmp_path
        / "provider.mp4"
    )

    adapter = FastVideoShotGenerator(
        provider,
        video_probe=FailingProbe(),
    )

    with pytest.raises(
        FastVideoShotGeneratorError,
        match="Could not inspect generated shot",
    ):
        asyncio.run(
            adapter.generate_shot(
                shot=_shot(),
                prompt="Tiger scene",
                output_path=(
                    tmp_path
                    / "job"
                    / "shots"
                    / "shot_001.mp4"
                ),
            )
        )
