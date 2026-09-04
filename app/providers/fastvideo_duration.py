from __future__ import annotations

import math
from dataclasses import dataclass


FASTWAN_22_TI2V_5B_ALIASES = {
    "fastwan2.2-ti2v-5b",
    "fastvideo/fastwan2.2-ti2v-5b-fullattn-diffusers",
}

FASTWAN_TEMPORAL_STRIDE = 4
FASTWAN_REQUIRED_FPS = 24


@dataclass(frozen=True, slots=True)
class ResolvedFastVideoDuration:
    """Provider-side duration decision for one FastVideo request."""

    requested_seconds: float
    fps: int
    requested_frames: int
    generation_frames: int | None
    generation_seconds: float | None
    uses_explicit_num_frames: bool


def is_fastwan22_ti2v_5b(model: str) -> bool:
    normalized = model.strip().lower()
    if normalized in FASTWAN_22_TI2V_5B_ALIASES:
        return True

    # Keep the short alias and full HF id robust to minor naming wrappers.
    return "fastwan2.2-ti2v-5b" in normalized


def resolve_fastvideo_duration(
    *,
    model: str,
    duration_seconds: float,
    fps: int,
) -> ResolvedFastVideoDuration:
    """
    Resolve semantic duration to the request shape expected by FastVideo.

    For FastWan2.2 TI2V 5B, generate the smallest model-compatible frame
    count that is *not shorter* than the requested duration.

    Wan's temporal grid is represented as 4n + 1 frames. For a 5.0 second
    request at 24 FPS:

        requested = ceil(5.0 * 24) = 120 frames
        aligned    = 121 frames
        generated  = 121 / 24 = 5.041666... seconds

    A later editing/trim stage can safely remove the small excess. This is
    preferable to allowing the runtime to produce 117 frames (4.875 s),
    because missing duration cannot be recovered without duplication or
    interpolation.

    Unknown FastVideo models retain the existing seconds-based behavior.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")

    if fps <= 0:
        raise ValueError("fps must be greater than zero.")

    requested_frames = max(
        1,
        int(math.ceil((duration_seconds * fps) - 1e-9)),
    )

    if not is_fastwan22_ti2v_5b(model):
        return ResolvedFastVideoDuration(
            requested_seconds=float(duration_seconds),
            fps=int(fps),
            requested_frames=requested_frames,
            generation_frames=None,
            generation_seconds=None,
            uses_explicit_num_frames=False,
        )

    if fps != FASTWAN_REQUIRED_FPS:
        raise ValueError(
            "FastWan2.2 TI2V 5B requires 24 FPS for the current "
            f"provider profile; got {fps}."
        )

    if requested_frames <= 1:
        generation_frames = 1
    else:
        # Smallest value of the form 4n + 1 that is >= requested_frames.
        latent_steps = math.ceil(
            (requested_frames - 1) / FASTWAN_TEMPORAL_STRIDE
        )
        generation_frames = (
            latent_steps * FASTWAN_TEMPORAL_STRIDE
        ) + 1

    generation_seconds = generation_frames / fps

    return ResolvedFastVideoDuration(
        requested_seconds=float(duration_seconds),
        fps=int(fps),
        requested_frames=requested_frames,
        generation_frames=generation_frames,
        generation_seconds=generation_seconds,
        uses_explicit_num_frames=True,
    )


__all__ = [
    "FASTWAN_22_TI2V_5B_ALIASES",
    "FASTWAN_REQUIRED_FPS",
    "FASTWAN_TEMPORAL_STRIDE",
    "ResolvedFastVideoDuration",
    "is_fastwan22_ti2v_5b",
    "resolve_fastvideo_duration",
]
