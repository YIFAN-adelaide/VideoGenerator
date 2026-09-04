from __future__ import annotations

from pathlib import Path

import pytest

from app.services.image_preprocessor import (
    ImagePreprocessor,
    resolve_aspect_ratio_canvas,
)
from app.services.image_probe import ImageProbeResult


class FakeImageProbe:
    def __init__(
        self,
        *,
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height

    async def probe(
        self,
        path: str | Path,
    ) -> ImageProbeResult:
        resolved = Path(path).resolve()

        return ImageProbeResult(
            path=resolved,
            width=self.width,
            height=self.height,
            aspect_ratio=(
                self.width / self.height
            ),
            pixel_format="rgb24",
        )


class FakeProcess:
    def __init__(
        self,
        output_path: Path,
    ) -> None:
        self.returncode = 0
        self._output_path = output_path

    async def communicate(
        self,
    ) -> tuple[bytes, bytes]:
        self._output_path.write_bytes(
            b"fake-prepared-png"
        )
        return b"", b""


def test_portrait_canvas_preserves_two_to_three_ratio():
    canvas = resolve_aspect_ratio_canvas(
        source_width=1024,
        source_height=1536,
        profile_width=1280,
        profile_height=704,
    )

    assert (
        canvas.width,
        canvas.height,
    ) == (768, 1152)

    assert canvas.aspect_ratio == pytest.approx(
        2 / 3
    )


def test_standard_16_9_image_keeps_16_9_ratio():
    canvas = resolve_aspect_ratio_canvas(
        source_width=1920,
        source_height=1080,
        profile_width=1280,
        profile_height=704,
    )

    assert (
        canvas.width,
        canvas.height,
    ) == (1280, 720)

    assert canvas.aspect_ratio == pytest.approx(
        16 / 9
    )


def test_existing_fastwan_canvas_stays_unchanged():
    canvas = resolve_aspect_ratio_canvas(
        source_width=1280,
        source_height=704,
        profile_width=1280,
        profile_height=704,
    )

    assert (
        canvas.width,
        canvas.height,
    ) == (1280, 704)


def test_panorama_preserves_ratio_with_smaller_pixel_budget():
    canvas = resolve_aspect_ratio_canvas(
        source_width=4000,
        source_height=1000,
        profile_width=1280,
        profile_height=704,
    )

    assert (
        canvas.width,
        canvas.height,
    ) == (1280, 320)

    assert canvas.aspect_ratio == pytest.approx(
        4.0
    )


@pytest.mark.asyncio
async def test_prepare_uses_contain_without_stretching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "portrait.jpg"
    source.write_bytes(b"fake-source")

    target = (
        tmp_path
        / "prepared"
        / "portrait.png"
    )

    seen_args: list[str] = []

    async def fake_create_subprocess_exec(
        *args,
        **kwargs,
    ):
        seen_args.extend(
            str(arg)
            for arg in args
        )

        return FakeProcess(
            target.resolve()
        )

    monkeypatch.setattr(
        (
            "app.services.image_preprocessor."
            "asyncio.create_subprocess_exec"
        ),
        fake_create_subprocess_exec,
    )

    processor = ImagePreprocessor(
        image_probe=FakeImageProbe(
            width=1024,
            height=1536,
        )
    )

    result = await processor.prepare(
        source,
        target,
        resolution="720p",
    )

    assert result.canvas_width == 768
    assert result.canvas_height == 1152
    assert result.canvas_aspect_ratio == pytest.approx(
        2 / 3
    )
    assert result.fit_mode == "contain"

    filter_arg = next(
        value
        for value in seen_args
        if value.startswith("scale=")
    )

    assert (
        "force_original_aspect_ratio=decrease"
        in filter_arg
    )
    assert "pad=768:1152" in filter_arg
