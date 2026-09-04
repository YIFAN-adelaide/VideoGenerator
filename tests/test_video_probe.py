from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.video_probe import (
    VideoProbe,
    VideoProbeError,
)


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_probe_reads_video_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "shot_001.mp4"
    video_path.write_bytes(b"fake-video")

    payload = {
        "streams": [
            {
                "width": 1280,
                "height": 704,
                "r_frame_rate": "24/1",
                "avg_frame_rate": "24/1",
                "nb_read_frames": "117",
                "duration": "4.875000",
            }
        ],
        "format": {
            "duration": "4.875000",
        },
    }

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess(
            stdout=json.dumps(payload).encode("utf-8"),
        )

    monkeypatch.setattr(
        "app.services.video_probe.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    result = await VideoProbe().probe(video_path)

    assert result.path == video_path.resolve()
    assert result.duration_seconds == pytest.approx(4.875)
    assert result.fps == pytest.approx(24.0)
    assert result.frame_count == 117
    assert result.width == 1280
    assert result.height == 704


@pytest.mark.asyncio
async def test_probe_uses_format_duration_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "shot.mp4"
    video_path.write_bytes(b"fake-video")

    payload = {
        "streams": [
            {
                "width": 1280,
                "height": 704,
                "r_frame_rate": "24/1",
                "avg_frame_rate": "24/1",
                "nb_read_frames": "121",
                "duration": "N/A",
            }
        ],
        "format": {
            "duration": "5.041667",
        },
    }

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess(
            stdout=json.dumps(payload).encode("utf-8"),
        )

    monkeypatch.setattr(
        "app.services.video_probe.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    result = await VideoProbe().probe(video_path)

    assert result.duration_seconds == pytest.approx(5.041667)
    assert result.frame_count == 121


@pytest.mark.asyncio
async def test_probe_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.mp4"

    with pytest.raises(
        VideoProbeError,
        match="Video does not exist",
    ):
        await VideoProbe().probe(missing_path)


@pytest.mark.asyncio
async def test_probe_surfaces_ffprobe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "broken.mp4"
    video_path.write_bytes(b"fake-video")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess(
            returncode=1,
            stderr=b"Invalid data found when processing input",
        )

    monkeypatch.setattr(
        "app.services.video_probe.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    with pytest.raises(
        VideoProbeError,
        match="Invalid data found",
    ):
        await VideoProbe().probe(video_path)


@pytest.mark.asyncio
async def test_probe_rejects_missing_video_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "audio_only.mp4"
    video_path.write_bytes(b"fake-video")

    payload = {
        "streams": [],
        "format": {
            "duration": "5.0",
        },
    }

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess(
            stdout=json.dumps(payload).encode("utf-8"),
        )

    monkeypatch.setattr(
        "app.services.video_probe.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    with pytest.raises(
        VideoProbeError,
        match="No video stream found",
    ):
        await VideoProbe().probe(video_path)
