from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from app.services.video_probe import VideoProbe


@dataclass(frozen=True, slots=True)
class FrameExtractionResult:
    """Result of extracting one exact frame from a video."""

    source_path: Path
    output_path: Path
    frame_index: int
    source_frame_count: int
    source_fps: float
    source_duration_seconds: float


class FrameExtractorError(RuntimeError):
    """Raised when an exact frame cannot be extracted."""


class FrameExtractor:
    """
    Extract exact video frames with ffmpeg.

    The extractor uses VideoProbe first so ``extract_last_frame`` can select
    the true final decoded frame by index instead of relying on approximate
    timestamp seeking such as ``-sseof``.

    Frame indexes are zero-based:

        121-frame clip -> last frame index = 120
    """

    def __init__(
        self,
        *,
        video_probe: VideoProbe | None = None,
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        ffmpeg_binary = ffmpeg_binary.strip()

        if not ffmpeg_binary:
            raise ValueError("ffmpeg_binary cannot be empty.")

        self._video_probe = video_probe or VideoProbe()
        self._ffmpeg_binary = ffmpeg_binary

    async def extract_last_frame(
        self,
        video_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = True,
    ) -> FrameExtractionResult:
        """
        Extract the final decoded frame from ``video_path`` as an image.

        The output format is inferred by ffmpeg from ``output_path``.
        PNG is recommended for continuity references because it is lossless.
        """
        source = self._validate_source(video_path)
        target = self._prepare_target(
            output_path,
            overwrite=overwrite,
        )

        try:
            metadata = await self._video_probe.probe(source)
        except Exception as exc:
            raise FrameExtractorError(
                f"Could not inspect source video {source}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if metadata.frame_count <= 0:
            raise FrameExtractorError(
                f"Source video has no decodable frames: {source}"
            )

        frame_index = metadata.frame_count - 1

        await self._extract_exact_frame(
            source=source,
            target=target,
            frame_index=frame_index,
            overwrite=overwrite,
        )

        return FrameExtractionResult(
            source_path=source,
            output_path=target,
            frame_index=frame_index,
            source_frame_count=metadata.frame_count,
            source_fps=metadata.fps,
            source_duration_seconds=metadata.duration_seconds,
        )

    async def extract_frame(
        self,
        video_path: str | Path,
        output_path: str | Path,
        *,
        frame_index: int,
        overwrite: bool = True,
    ) -> FrameExtractionResult:
        """Extract one zero-based frame index from a video."""
        source = self._validate_source(video_path)
        target = self._prepare_target(
            output_path,
            overwrite=overwrite,
        )

        try:
            metadata = await self._video_probe.probe(source)
        except Exception as exc:
            raise FrameExtractorError(
                f"Could not inspect source video {source}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if frame_index < 0:
            raise FrameExtractorError(
                "frame_index must be greater than or equal to zero."
            )

        if frame_index >= metadata.frame_count:
            raise FrameExtractorError(
                f"frame_index {frame_index} is outside source frame range "
                f"0..{metadata.frame_count - 1}."
            )

        await self._extract_exact_frame(
            source=source,
            target=target,
            frame_index=frame_index,
            overwrite=overwrite,
        )

        return FrameExtractionResult(
            source_path=source,
            output_path=target,
            frame_index=frame_index,
            source_frame_count=metadata.frame_count,
            source_fps=metadata.fps,
            source_duration_seconds=metadata.duration_seconds,
        )

    async def _extract_exact_frame(
        self,
        *,
        source: Path,
        target: Path,
        frame_index: int,
        overwrite: bool,
    ) -> None:
        # ``select=eq(n\,N)`` selects the exact decoded frame by zero-based
        # frame number. This avoids timestamp/keyframe ambiguity.
        select_filter = f"select=eq(n\\,{frame_index})"

        command = [
            self._ffmpeg_binary,
            "-v",
            "error",
            "-y" if overwrite else "-n",
            "-i",
            str(source),
            "-vf",
            select_filter,
            "-frames:v",
            "1",
            "-fps_mode",
            "vfr",
            str(target),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise FrameExtractorError(
                f"ffmpeg executable was not found: "
                f"{self._ffmpeg_binary}"
            ) from exc

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            message = stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()

            raise FrameExtractorError(
                message
                or (
                    f"ffmpeg failed to extract frame {frame_index} "
                    f"from {source}"
                )
            )

        if not target.exists():
            raise FrameExtractorError(
                f"ffmpeg completed but did not create output: {target}"
            )

        if target.stat().st_size <= 0:
            raise FrameExtractorError(
                f"Extracted frame is empty: {target}"
            )

    @staticmethod
    def _validate_source(
        video_path: str | Path,
    ) -> Path:
        source = Path(video_path).expanduser().resolve()

        if not source.exists():
            raise FrameExtractorError(
                f"Source video does not exist: {source}"
            )

        if not source.is_file():
            raise FrameExtractorError(
                f"Source video path is not a file: {source}"
            )

        if source.stat().st_size <= 0:
            raise FrameExtractorError(
                f"Source video is empty: {source}"
            )

        return source

    @staticmethod
    def _prepare_target(
        output_path: str | Path,
        *,
        overwrite: bool,
    ) -> Path:
        target = Path(output_path).expanduser().resolve()

        if not target.suffix:
            raise FrameExtractorError(
                "output_path must include an image file extension."
            )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if target.exists() and not overwrite:
            raise FrameExtractorError(
                f"Output already exists: {target}"
            )

        return target


__all__ = [
    "FrameExtractionResult",
    "FrameExtractor",
    "FrameExtractorError",
]
