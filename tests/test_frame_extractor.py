from __future__ import annotations

from pathlib import Path

import pytest

from app.services.frame_extractor import (
    FrameExtractor,
    FrameExtractorError,
)
from app.services.video_probe import VideoProbeResult


class FakeVideoProbe:
    def __init__(
        self,
        *,
        duration_seconds: float = 5.041667,
        fps: float = 24.0,
        frame_count: int = 121,
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


class FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
        output_path: Path | None = None,
    ) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._output_path = output_path

    async def communicate(
        self,
    ) -> tuple[bytes, bytes]:
        if (
            self.returncode == 0
            and self._output_path is not None
        ):
            self._output_path.write_bytes(
                b"fake-png"
            )

        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_extract_last_frame_uses_exact_final_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "shot_001.mp4"
    source.write_bytes(b"fake-video")

    target = tmp_path / "refs" / "shot_001_last.png"

    seen_args: list[str] = []

    async def fake_create_subprocess_exec(
        *args,
        **kwargs,
    ):
        seen_args.extend(str(arg) for arg in args)

        return FakeProcess(
            output_path=target.resolve(),
        )

    monkeypatch.setattr(
        (
            "app.services.frame_extractor."
            "asyncio.create_subprocess_exec"
        ),
        fake_create_subprocess_exec,
    )

    extractor = FrameExtractor(
        video_probe=FakeVideoProbe(
            frame_count=121,
        )
    )

    result = await extractor.extract_last_frame(
        source,
        target,
    )

    assert result.frame_index == 120
    assert result.source_frame_count == 121
    assert result.source_fps == pytest.approx(24.0)
    assert result.source_duration_seconds == pytest.approx(
        5.041667
    )
    assert result.output_path == target.resolve()
    assert target.exists()

    assert "select=eq(n\\,120)" in seen_args
    assert "-frames:v" in seen_args
    assert "1" in seen_args


@pytest.mark.asyncio
async def test_extract_specific_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "shot.mp4"
    source.write_bytes(b"fake-video")

    target = tmp_path / "frame_024.png"

    seen_args: list[str] = []

    async def fake_create_subprocess_exec(
        *args,
        **kwargs,
    ):
        seen_args.extend(str(arg) for arg in args)

        return FakeProcess(
            output_path=target.resolve(),
        )

    monkeypatch.setattr(
        (
            "app.services.frame_extractor."
            "asyncio.create_subprocess_exec"
        ),
        fake_create_subprocess_exec,
    )

    extractor = FrameExtractor(
        video_probe=FakeVideoProbe(
            frame_count=121,
        )
    )

    result = await extractor.extract_frame(
        source,
        target,
        frame_index=24,
    )

    assert result.frame_index == 24
    assert "select=eq(n\\,24)" in seen_args


@pytest.mark.asyncio
async def test_extract_frame_rejects_out_of_range_index(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shot.mp4"
    source.write_bytes(b"fake-video")

    extractor = FrameExtractor(
        video_probe=FakeVideoProbe(
            frame_count=121,
        )
    )

    with pytest.raises(
        FrameExtractorError,
        match="outside source frame range",
    ):
        await extractor.extract_frame(
            source,
            tmp_path / "frame.png",
            frame_index=121,
        )


@pytest.mark.asyncio
async def test_extract_last_frame_rejects_missing_source(
    tmp_path: Path,
) -> None:
    extractor = FrameExtractor(
        video_probe=FakeVideoProbe()
    )

    with pytest.raises(
        FrameExtractorError,
        match="does not exist",
    ):
        await extractor.extract_last_frame(
            tmp_path / "missing.mp4",
            tmp_path / "frame.png",
        )


@pytest.mark.asyncio
async def test_extract_last_frame_surfaces_ffmpeg_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "shot.mp4"
    source.write_bytes(b"fake-video")

    target = tmp_path / "last.png"

    async def fake_create_subprocess_exec(
        *args,
        **kwargs,
    ):
        return FakeProcess(
            returncode=1,
            stderr=b"simulated ffmpeg failure",
        )

    monkeypatch.setattr(
        (
            "app.services.frame_extractor."
            "asyncio.create_subprocess_exec"
        ),
        fake_create_subprocess_exec,
    )

    extractor = FrameExtractor(
        video_probe=FakeVideoProbe()
    )

    with pytest.raises(
        FrameExtractorError,
        match="simulated ffmpeg failure",
    ):
        await extractor.extract_last_frame(
            source,
            target,
        )


@pytest.mark.asyncio
async def test_extract_last_frame_rejects_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shot.mp4"
    source.write_bytes(b"fake-video")

    target = tmp_path / "last.png"
    target.write_bytes(b"existing")

    extractor = FrameExtractor(
        video_probe=FakeVideoProbe()
    )

    with pytest.raises(
        FrameExtractorError,
        match="Output already exists",
    ):
        await extractor.extract_last_frame(
            source,
            target,
            overwrite=False,
        )
