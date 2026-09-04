from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class VideoComposerError(RuntimeError):
    """Base exception raised by the video-composition service."""


class VideoComposerConfigurationError(VideoComposerError):
    """Raised when FFmpeg or composer configuration is invalid."""


class VideoCompositionError(VideoComposerError):
    """Raised when FFmpeg cannot create the final video."""


@dataclass(frozen=True, slots=True)
class CompositionResult:
    output_path: Path
    input_count: int
    elapsed_seconds: float
    file_size_bytes: int


class VideoComposer:
    """
    Join independently generated MP4 shots into one final MP4.

    V1 uses FFmpeg's concat demuxer with stream copying:
    - no re-encoding
    - no quality loss
    - very low CPU overhead

    Input clips must be stream-compatible.
    """

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        overwrite: bool = True,
    ) -> None:
        if not ffmpeg_binary.strip():
            raise VideoComposerConfigurationError(
                "ffmpeg_binary cannot be empty."
            )

        self._ffmpeg_binary = ffmpeg_binary
        self._overwrite = overwrite

    @property
    def ffmpeg_binary(self) -> str:
        return self._ffmpeg_binary

    def ensure_available(self) -> None:
        candidate = Path(self._ffmpeg_binary)

        if candidate.exists() and candidate.is_file():
            return

        if shutil.which(self._ffmpeg_binary) is None:
            raise VideoComposerConfigurationError(
                f"FFmpeg executable '{self._ffmpeg_binary}' was not found. "
                "Install FFmpeg or configure ffmpeg_binary with its full path."
            )

    async def concatenate(
        self,
        shot_paths: Sequence[str | Path],
        output_path: str | Path,
    ) -> CompositionResult:
        inputs = [Path(path).resolve() for path in shot_paths]

        if not inputs:
            raise VideoCompositionError(
                "At least one shot is required for composition."
            )

        for path in inputs:
            if not path.exists():
                raise VideoCompositionError(
                    f"Input shot does not exist: {path}"
                )

            if not path.is_file():
                raise VideoCompositionError(
                    f"Input shot is not a file: {path}"
                )

            if path.stat().st_size <= 0:
                raise VideoCompositionError(
                    f"Input shot is empty: {path}"
                )

        output = Path(output_path).resolve()

        if output in inputs:
            raise VideoCompositionError(
                "output_path cannot be one of the input shot paths."
            )

        output.parent.mkdir(parents=True, exist_ok=True)

        self.ensure_available()

        started = time.perf_counter()
        manifest_path: Path | None = None

        try:
            manifest_path = self._write_concat_manifest(
                inputs,
                directory=output.parent,
            )

            await self._run_ffmpeg(
                manifest_path=manifest_path,
                output_path=output,
            )

        finally:
            if manifest_path is not None:
                manifest_path.unlink(missing_ok=True)

        if not output.exists():
            raise VideoCompositionError(
                "FFmpeg finished without creating the expected output file: "
                f"{output}"
            )

        file_size = output.stat().st_size

        if file_size <= 0:
            raise VideoCompositionError(
                f"FFmpeg created an empty output file: {output}"
            )

        return CompositionResult(
            output_path=output,
            input_count=len(inputs),
            elapsed_seconds=time.perf_counter() - started,
            file_size_bytes=file_size,
        )

    def _write_concat_manifest(
        self,
        inputs: Sequence[Path],
        *,
        directory: Path,
    ) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".ffconcat.txt",
            prefix="video_composer_",
            dir=directory,
            delete=False,
            newline="\n",
        ) as handle:
            for path in inputs:
                safe_path = self._escape_concat_path(path)
                handle.write(f"file '{safe_path}'\n")

            return Path(handle.name)

    @staticmethod
    def _escape_concat_path(path: Path) -> str:
        normalized = path.as_posix()
        return normalized.replace("'", r"'\''")

    async def _run_ffmpeg(
        self,
        *,
        manifest_path: Path,
        output_path: Path,
    ) -> None:
        command = [
            self._ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-y" if self._overwrite else "-n",
            str(output_path),
        ]

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            stderr_text = stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()
            stdout_text = stdout.decode(
                "utf-8",
                errors="replace",
            ).strip()

            diagnostic = stderr_text or stdout_text or (
                f"FFmpeg exited with code {process.returncode}."
            )

            raise VideoCompositionError(
                f"FFmpeg composition failed: {diagnostic}"
            )
