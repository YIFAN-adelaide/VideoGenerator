from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.services.video_composer import (
    CompositionResult,
    VideoComposer,
    VideoComposerConfigurationError,
    VideoCompositionError,
)


def _make_nonempty_file(
    path: Path,
    content: bytes = b"fake-video",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_concatenate_rejects_empty_input_list(tmp_path: Path):
    composer = VideoComposer()

    with pytest.raises(
        VideoCompositionError,
        match="At least one shot",
    ):
        asyncio.run(
            composer.concatenate(
                [],
                tmp_path / "final.mp4",
            )
        )


def test_concatenate_rejects_missing_input(tmp_path: Path):
    composer = VideoComposer()

    missing = tmp_path / "missing.mp4"

    with pytest.raises(
        VideoCompositionError,
        match="does not exist",
    ):
        asyncio.run(
            composer.concatenate(
                [missing],
                tmp_path / "final.mp4",
            )
        )


def test_concatenate_rejects_output_as_input(tmp_path: Path):
    shot = _make_nonempty_file(
        tmp_path / "same.mp4"
    )

    composer = VideoComposer()

    with pytest.raises(
        VideoCompositionError,
        match="output_path cannot",
    ):
        asyncio.run(
            composer.concatenate(
                [shot],
                shot,
            )
        )


def test_manifest_preserves_input_order(tmp_path: Path):
    composer = VideoComposer()

    shot_1 = _make_nonempty_file(
        tmp_path / "shot_001.mp4"
    )
    shot_2 = _make_nonempty_file(
        tmp_path / "shot_002.mp4"
    )
    shot_3 = _make_nonempty_file(
        tmp_path / "shot_003.mp4"
    )

    manifest = composer._write_concat_manifest(
        [
            shot_1.resolve(),
            shot_2.resolve(),
            shot_3.resolve(),
        ],
        directory=tmp_path,
    )

    try:
        contents = manifest.read_text(
            encoding="utf-8"
        )
        lines = contents.splitlines()

        assert "shot_001.mp4" in lines[0]
        assert "shot_002.mp4" in lines[1]
        assert "shot_003.mp4" in lines[2]

    finally:
        manifest.unlink(missing_ok=True)


def test_concatenate_returns_result_without_real_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    shot_1 = _make_nonempty_file(
        tmp_path / "shot_001.mp4"
    )
    shot_2 = _make_nonempty_file(
        tmp_path / "shot_002.mp4"
    )
    output = tmp_path / "final.mp4"

    composer = VideoComposer()

    monkeypatch.setattr(
        composer,
        "ensure_available",
        lambda: None,
    )

    async def fake_run_ffmpeg(
        *,
        manifest_path: Path,
        output_path: Path,
    ) -> None:
        assert manifest_path.exists()
        output_path.write_bytes(
            b"composed-video"
        )

    monkeypatch.setattr(
        composer,
        "_run_ffmpeg",
        fake_run_ffmpeg,
    )

    result = asyncio.run(
        composer.concatenate(
            [shot_1, shot_2],
            output,
        )
    )

    assert isinstance(
        result,
        CompositionResult,
    )
    assert result.output_path == output.resolve()
    assert result.input_count == 2
    assert result.file_size_bytes == len(
        b"composed-video"
    )
    assert result.elapsed_seconds >= 0


def test_manifest_is_removed_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    shot = _make_nonempty_file(
        tmp_path / "shot_001.mp4"
    )
    output = tmp_path / "final.mp4"

    composer = VideoComposer()

    monkeypatch.setattr(
        composer,
        "ensure_available",
        lambda: None,
    )

    captured_manifest: Path | None = None

    async def fake_run_ffmpeg(
        *,
        manifest_path: Path,
        output_path: Path,
    ) -> None:
        nonlocal captured_manifest
        captured_manifest = manifest_path
        assert manifest_path.exists()
        output_path.write_bytes(b"video")

    monkeypatch.setattr(
        composer,
        "_run_ffmpeg",
        fake_run_ffmpeg,
    )

    asyncio.run(
        composer.concatenate(
            [shot],
            output,
        )
    )

    assert captured_manifest is not None
    assert not captured_manifest.exists()


def test_ensure_available_raises_for_missing_binary(
    monkeypatch: pytest.MonkeyPatch,
):
    composer = VideoComposer(
        ffmpeg_binary="definitely-not-real-ffmpeg",
    )

    monkeypatch.setattr(
        "app.services.video_composer.shutil.which",
        lambda _: None,
    )

    with pytest.raises(
        VideoComposerConfigurationError,
        match="was not found",
    ):
        composer.ensure_available()
