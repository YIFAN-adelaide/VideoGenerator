from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VideoProbeResult:
    """Observed properties of a generated video file."""

    path: Path
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


class VideoProbeError(RuntimeError):
    """Raised when ffprobe cannot inspect a video successfully."""
    pass


class VideoProbe:
    """Inspect generated video files using ffprobe."""

    def __init__(self, ffprobe_binary: str = "ffprobe") -> None:
        ffprobe_binary = ffprobe_binary.strip()
        if not ffprobe_binary:
            raise ValueError("ffprobe_binary cannot be empty.")
        self._ffprobe_binary = ffprobe_binary

    async def probe(self, path: str | Path) -> VideoProbeResult:
        video_path = Path(path).expanduser().resolve()

        if not video_path.exists():
            raise VideoProbeError(f"Video does not exist: {video_path}")
        if not video_path.is_file():
            raise VideoProbeError(f"Video path is not a file: {video_path}")
        if video_path.stat().st_size <= 0:
            raise VideoProbeError(f"Video file is empty: {video_path}")

        process = await asyncio.create_subprocess_exec(
            self._ffprobe_binary,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream="
                "width,height,"
                "r_frame_rate,"
                "avg_frame_rate,"
                "nb_read_frames,"
                "nb_frames,"
                "duration:"
                "format=duration"
            ),
            "-of",
            "json",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise VideoProbeError(
                message or f"ffprobe failed for: {video_path}"
            )

        try:
            payload = json.loads(
                stdout.decode("utf-8", errors="replace")
            )
        except json.JSONDecodeError as exc:
            raise VideoProbeError(
                f"ffprobe returned invalid JSON for {video_path}: {exc}"
            ) from exc

        return self._parse_result(path=video_path, payload=payload)

    def _parse_result(
        self,
        *,
        path: Path,
        payload: dict[str, Any],
    ) -> VideoProbeResult:
        streams = payload.get("streams")
        if not isinstance(streams, list) or not streams:
            raise VideoProbeError(f"No video stream found: {path}")

        stream = streams[0]
        if not isinstance(stream, dict):
            raise VideoProbeError(
                f"Invalid video stream metadata: {path}"
            )

        width = self._positive_int(
            stream.get("width"),
            field_name="width",
            path=path,
        )
        height = self._positive_int(
            stream.get("height"),
            field_name="height",
            path=path,
        )
        fps = self._read_fps(stream=stream, path=path)
        frame_count = self._read_frame_count(
            stream=stream,
            path=path,
        )
        duration_seconds = self._read_duration(
            stream=stream,
            payload=payload,
            path=path,
        )

        return VideoProbeResult(
            path=path,
            duration_seconds=duration_seconds,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )

    def _read_fps(
        self,
        *,
        stream: dict[str, Any],
        path: Path,
    ) -> float:
        for key in ("r_frame_rate", "avg_frame_rate"):
            value = stream.get(key)
            if (
                isinstance(value, str)
                and value.strip()
                and value != "0/0"
            ):
                fps = self._parse_fraction(value)
                if fps > 0:
                    return fps

        raise VideoProbeError(f"Could not determine FPS: {path}")

    def _read_frame_count(
        self,
        *,
        stream: dict[str, Any],
        path: Path,
    ) -> int:
        for key in ("nb_read_frames", "nb_frames"):
            value = stream.get(key)
            if value in (None, "", "N/A"):
                continue

            try:
                frame_count = int(value)
            except (TypeError, ValueError):
                continue

            if frame_count > 0:
                return frame_count

        raise VideoProbeError(
            f"Could not determine frame count: {path}"
        )

    def _read_duration(
        self,
        *,
        stream: dict[str, Any],
        payload: dict[str, Any],
        path: Path,
    ) -> float:
        candidates: list[Any] = [stream.get("duration")]

        format_info = payload.get("format")
        if isinstance(format_info, dict):
            candidates.append(format_info.get("duration"))

        for value in candidates:
            if value in (None, "", "N/A"):
                continue

            try:
                duration = float(value)
            except (TypeError, ValueError):
                continue

            if duration > 0:
                return duration

        raise VideoProbeError(
            f"Could not determine duration: {path}"
        )

    @staticmethod
    def _parse_fraction(value: str) -> float:
        if "/" not in value:
            try:
                return float(value)
            except ValueError as exc:
                raise VideoProbeError(
                    f"Invalid frame rate: {value}"
                ) from exc

        numerator_text, denominator_text = value.split(
            "/",
            maxsplit=1,
        )

        try:
            numerator = float(numerator_text)
            denominator = float(denominator_text)
        except ValueError as exc:
            raise VideoProbeError(
                f"Invalid frame rate: {value}"
            ) from exc

        if denominator == 0:
            raise VideoProbeError(
                f"Invalid frame rate denominator: {value}"
            )

        return numerator / denominator

    @staticmethod
    def _positive_int(
        value: Any,
        *,
        field_name: str,
        path: Path,
    ) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise VideoProbeError(
                f"Invalid {field_name} for {path}: {value!r}"
            ) from exc

        if result <= 0:
            raise VideoProbeError(
                f"Invalid {field_name} for {path}: {result}"
            )

        return result


__all__ = [
    "VideoProbe",
    "VideoProbeError",
    "VideoProbeResult",
]
