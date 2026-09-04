from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path

from app.services.image_probe import ImageProbe


DEFAULT_VIDEO_PROFILES: dict[str, tuple[int, int]] = {
    "480p": (832, 480),
    "720p": (1280, 704),
}


@dataclass(frozen=True, slots=True)
class ImageCanvas:
    """Model-ready canvas chosen while preserving source aspect ratio."""

    width: int
    height: int
    aspect_ratio: float
    source_aspect_ratio: float
    target_pixels: int

    @property
    def aspect_ratio_error(self) -> float:
        return abs(
            self.aspect_ratio
            - self.source_aspect_ratio
        )


@dataclass(frozen=True, slots=True)
class ImagePreparationResult:
    """Prepared image plus the canvas that video generation should use."""

    source_path: Path
    output_path: Path

    source_width: int
    source_height: int
    source_aspect_ratio: float

    canvas_width: int
    canvas_height: int
    canvas_aspect_ratio: float

    fit_mode: str = "contain"


class ImagePreprocessorError(RuntimeError):
    """Raised when a model-ready image cannot be prepared."""


def resolve_aspect_ratio_canvas(
    *,
    source_width: int,
    source_height: int,
    profile_width: int,
    profile_height: int,
    alignment: int = 16,
    min_dimension: int = 128,
) -> ImageCanvas:
    """
    Choose a model-ready canvas with nearly the same aspect ratio.

    The selected canvas:
      * keeps the source width/height proportion as closely as possible,
      * uses dimensions aligned to ``alignment``,
      * stays within the profile's largest dimension,
      * aims for roughly the same pixel budget as the profile.

    Examples with the 720p profile (1280x704):
      1024x1536  -> 768x1152  (2:3 portrait preserved)
      1920x1080  -> 1280x720  (16:9 preserved)
      1280x704   -> 1280x704  (unchanged)
      4000x1000  -> 1280x320  (4:1 preserved)

    The preprocessor still uses ``contain`` when rendering, so even if an
    unusual ratio cannot be represented exactly on the aligned grid, pixels
    are never stretched or squashed.
    """
    for name, value in (
        ("source_width", source_width),
        ("source_height", source_height),
        ("profile_width", profile_width),
        ("profile_height", profile_height),
        ("alignment", alignment),
        ("min_dimension", min_dimension),
    ):
        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

    if min_dimension % alignment != 0:
        raise ValueError(
            "min_dimension must be divisible by alignment."
        )

    source_ratio = source_width / source_height
    target_pixels = profile_width * profile_height
    max_dimension = max(
        profile_width,
        profile_height,
    )

    candidates: list[
        tuple[
            float,
            float,
            float,
            int,
            int,
        ]
    ] = []

    def add_candidate(
        width: int,
        height: int,
    ) -> None:
        if (
            width < min_dimension
            or height < min_dimension
            or width > max_dimension
            or height > max_dimension
        ):
            return

        ratio = width / height
        pixels = width * height

        # Log-space errors behave symmetrically for portrait/landscape.
        aspect_error = abs(
            math.log(ratio / source_ratio)
        )
        area_error = abs(
            math.log(pixels / target_pixels)
        )

        # Aspect ratio matters much more than exact pixel budget.
        score = (10.0 * aspect_error) + area_error

        candidates.append(
            (
                score,
                aspect_error,
                area_error,
                width,
                height,
            )
        )

    aligned_values = range(
        min_dimension,
        max_dimension + 1,
        alignment,
    )

    for width in aligned_values:
        estimated_height = width / source_ratio
        height = int(
            round(
                estimated_height / alignment
            )
        ) * alignment
        add_candidate(width, height)

    for height in aligned_values:
        estimated_width = height * source_ratio
        width = int(
            round(
                estimated_width / alignment
            )
        ) * alignment
        add_candidate(width, height)

    if not candidates:
        raise ImagePreprocessorError(
            "Could not find a model-ready canvas for "
            f"{source_width}x{source_height}. The image "
            "aspect ratio is too extreme for the current "
            "generation profile."
        )

    _, _, _, width, height = min(candidates)

    return ImageCanvas(
        width=width,
        height=height,
        aspect_ratio=width / height,
        source_aspect_ratio=source_ratio,
        target_pixels=target_pixels,
    )


class ImagePreprocessor:
    """
    Prepare user/reference images without changing their visual proportions.

    The original image is never modified.

    The generated copy is placed on an aligned canvas whose aspect ratio is
    chosen to closely match the source. ``contain`` is then used so the source
    pixels are never stretched. Any padding should normally be only a few
    pixels because the canvas itself follows the source ratio.
    """

    def __init__(
        self,
        *,
        image_probe: ImageProbe | None = None,
        ffmpeg_binary: str = "ffmpeg",
        profiles: dict[str, tuple[int, int]] | None = None,
        alignment: int = 16,
        min_dimension: int = 128,
    ) -> None:
        ffmpeg_binary = ffmpeg_binary.strip()

        if not ffmpeg_binary:
            raise ValueError("ffmpeg_binary cannot be empty.")

        if alignment <= 0:
            raise ValueError(
                "alignment must be greater than zero."
            )

        self._image_probe = image_probe or ImageProbe()
        self._ffmpeg_binary = ffmpeg_binary
        self._profiles = dict(
            profiles or DEFAULT_VIDEO_PROFILES
        )
        self._alignment = alignment
        self._min_dimension = min_dimension

    async def prepare(
        self,
        source_path: str | Path,
        output_path: str | Path,
        *,
        resolution: str = "720p",
        overwrite: bool = True,
    ) -> ImagePreparationResult:
        source = self._validate_source(source_path)
        target = self._prepare_target(
            output_path,
            overwrite=overwrite,
        )

        try:
            source_info = await self._image_probe.probe(
                source
            )
        except Exception as exc:
            raise ImagePreprocessorError(
                f"Could not inspect input image {source}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            profile_width, profile_height = (
                self._profiles[resolution]
            )
        except KeyError as exc:
            raise ImagePreprocessorError(
                f"Unsupported generation profile: "
                f"{resolution!r}"
            ) from exc

        canvas = resolve_aspect_ratio_canvas(
            source_width=source_info.width,
            source_height=source_info.height,
            profile_width=profile_width,
            profile_height=profile_height,
            alignment=self._alignment,
            min_dimension=self._min_dimension,
        )

        await self._render_contain(
            source=source,
            target=target,
            canvas=canvas,
            overwrite=overwrite,
        )

        return ImagePreparationResult(
            source_path=source,
            output_path=target,
            source_width=source_info.width,
            source_height=source_info.height,
            source_aspect_ratio=(
                source_info.aspect_ratio
            ),
            canvas_width=canvas.width,
            canvas_height=canvas.height,
            canvas_aspect_ratio=canvas.aspect_ratio,
            fit_mode="contain",
        )

    async def _render_contain(
        self,
        *,
        source: Path,
        target: Path,
        canvas: ImageCanvas,
        overwrite: bool,
    ) -> None:
        # The canvas has already been selected to be very close to the source
        # ratio. ``contain`` prevents distortion; pad only fills any tiny
        # alignment remainder.
        video_filter = (
            f"scale={canvas.width}:{canvas.height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={canvas.width}:{canvas.height}:"
            "(ow-iw)/2:(oh-ih)/2:color=black"
        )

        command = [
            self._ffmpeg_binary,
            "-v",
            "error",
            "-y" if overwrite else "-n",
            "-i",
            str(source),
            "-vf",
            video_filter,
            "-frames:v",
            "1",
            str(target),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ImagePreprocessorError(
                f"ffmpeg executable was not found: "
                f"{self._ffmpeg_binary}"
            ) from exc

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            message = stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()

            raise ImagePreprocessorError(
                message
                or f"ffmpeg could not prepare image: {source}"
            )

        if not target.exists():
            raise ImagePreprocessorError(
                f"ffmpeg completed but did not create: "
                f"{target}"
            )

        if target.stat().st_size <= 0:
            raise ImagePreprocessorError(
                f"Prepared image is empty: {target}"
            )

    @staticmethod
    def _validate_source(
        source_path: str | Path,
    ) -> Path:
        source = Path(source_path).expanduser().resolve()

        if not source.exists():
            raise ImagePreprocessorError(
                f"Input image does not exist: {source}"
            )

        if not source.is_file():
            raise ImagePreprocessorError(
                f"Input image path is not a file: {source}"
            )

        if source.stat().st_size <= 0:
            raise ImagePreprocessorError(
                f"Input image is empty: {source}"
            )

        return source

    @staticmethod
    def _prepare_target(
        output_path: str | Path,
        *,
        overwrite: bool,
    ) -> Path:
        target = Path(output_path).expanduser().resolve()

        if target.suffix.lower() != ".png":
            raise ImagePreprocessorError(
                "Prepared reference images must use a .png "
                "output path."
            )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if target.exists() and not overwrite:
            raise ImagePreprocessorError(
                f"Output already exists: {target}"
            )

        return target


__all__ = [
    "DEFAULT_VIDEO_PROFILES",
    "ImageCanvas",
    "ImagePreparationResult",
    "ImagePreprocessor",
    "ImagePreprocessorError",
    "resolve_aspect_ratio_canvas",
]
