from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageProbeResult:
    """Observed properties of one input image."""

    path: Path
    width: int
    height: int
    aspect_ratio: float
    pixel_format: str | None = None


class ImageProbeError(RuntimeError):
    """Raised when ffprobe cannot inspect an image successfully."""


class ImageProbe:
    """Inspect an image using the already-required ffprobe runtime."""

    def __init__(self, ffprobe_binary: str = "ffprobe") -> None:
        ffprobe_binary = ffprobe_binary.strip()

        if not ffprobe_binary:
            raise ValueError("ffprobe_binary cannot be empty.")

        self._ffprobe_binary = ffprobe_binary

    async def probe(
        self,
        path: str | Path,
    ) -> ImageProbeResult:
        image_path = Path(path).expanduser().resolve()

        if not image_path.exists():
            raise ImageProbeError(
                f"Image does not exist: {image_path}"
            )

        if not image_path.is_file():
            raise ImageProbeError(
                f"Image path is not a file: {image_path}"
            )

        if image_path.stat().st_size <= 0:
            raise ImageProbeError(
                f"Image file is empty: {image_path}"
            )

        try:
            process = await asyncio.create_subprocess_exec(
                self._ffprobe_binary,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,pix_fmt",
                "-of",
                "json",
                str(image_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ImageProbeError(
                f"ffprobe executable was not found: "
                f"{self._ffprobe_binary}"
            ) from exc

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            message = stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()

            raise ImageProbeError(
                message
                or f"ffprobe failed for: {image_path}"
            )

        try:
            payload = json.loads(
                stdout.decode(
                    "utf-8",
                    errors="replace",
                )
            )
        except json.JSONDecodeError as exc:
            raise ImageProbeError(
                f"ffprobe returned invalid JSON for "
                f"{image_path}: {exc}"
            ) from exc

        return self._parse_result(
            image_path=image_path,
            payload=payload,
        )

    @staticmethod
    def _parse_result(
        *,
        image_path: Path,
        payload: dict[str, Any],
    ) -> ImageProbeResult:
        streams = payload.get("streams")

        if not isinstance(streams, list) or not streams:
            raise ImageProbeError(
                f"No image/video stream found: {image_path}"
            )

        stream = streams[0]

        if not isinstance(stream, dict):
            raise ImageProbeError(
                f"Invalid image stream metadata: {image_path}"
            )

        try:
            width = int(stream["width"])
            height = int(stream["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ImageProbeError(
                f"Could not determine image dimensions: "
                f"{image_path}"
            ) from exc

        if width <= 0 or height <= 0:
            raise ImageProbeError(
                f"Invalid image dimensions "
                f"{width}x{height}: {image_path}"
            )

        pixel_format = stream.get("pix_fmt")

        return ImageProbeResult(
            path=image_path,
            width=width,
            height=height,
            aspect_ratio=width / height,
            pixel_format=(
                str(pixel_format)
                if pixel_format is not None
                else None
            ),
        )


__all__ = [
    "ImageProbe",
    "ImageProbeError",
    "ImageProbeResult",
]
