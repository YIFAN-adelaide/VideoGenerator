from __future__ import annotations

import math
import secrets
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


HELIOS_FRAME_CHUNK = 33
MAX_SEED = 2**31 - 1


def normalize_helios_frame_count(
    duration_seconds: float,
    fps: int,
) -> tuple[int, int, float]:
    """
    Convert a user-facing duration into a Helios-compatible frame count.

    Helios generates autoregressively in 33-frame chunks. We round upward
    rather than downward so the generated video is never shorter than the
    requested duration.

    Returns:
        requested_frames,
        normalized_frames,
        actual_duration_seconds
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    requested_frames = max(1, math.ceil(duration_seconds * fps))
    normalized_frames = (
        math.ceil(requested_frames / HELIOS_FRAME_CHUNK)
        * HELIOS_FRAME_CHUNK
    )
    actual_duration_seconds = normalized_frames / fps

    return (
        requested_frames,
        normalized_frames,
        actual_duration_seconds,
    )


def resolve_seed(seed: int | None) -> int:
    """Preserve an explicit seed or generate one that can be returned in metadata."""
    if seed is None:
        return secrets.randbelow(MAX_SEED + 1)

    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")

    return seed


class HeliosGenerationParams(BaseModel):
    """
    Fully resolved parameters for one Helios text-to-video inference.

    This is an internal model. The public API should remain generic and should
    not expose Helios-specific tuning knobs directly.
    """

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None

    profile_name: str
    requested_resolution: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    requested_duration_seconds: float = Field(gt=0)
    fps: int = Field(gt=0)

    requested_frames: int = Field(gt=0)
    num_frames: int = Field(gt=0)
    actual_duration_seconds: float = Field(gt=0)

    seed: int = Field(ge=0, le=MAX_SEED)

    guidance_scale: float = Field(ge=0)
    pyramid_num_inference_steps_list: tuple[int, ...]
    amplify_first_chunk: bool = True

    # Used by the repository-native HeliosPipeline. The newer
    # HeliosPyramidPipeline does not require this flag, so the provider only
    # forwards it when the loaded pipeline supports it.
    enable_stage2: bool = True

    history_sizes: tuple[int, ...] = (16, 2, 1)
    num_latent_frames_per_chunk: int = Field(default=9, gt=0)
    keep_first_frame: bool = True

    output_type: Literal["np"] = "np"

    @model_validator(mode="after")
    def validate_helios_constraints(self) -> "HeliosGenerationParams":
        if self.num_frames % HELIOS_FRAME_CHUNK != 0:
            raise ValueError(
                f"num_frames must be a multiple of {HELIOS_FRAME_CHUNK}"
            )

        if not self.pyramid_num_inference_steps_list:
            raise ValueError(
                "pyramid_num_inference_steps_list cannot be empty"
            )

        if any(step < 1 for step in self.pyramid_num_inference_steps_list):
            raise ValueError("all pyramid inference steps must be >= 1")

        return self
