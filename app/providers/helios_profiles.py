from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeliosGenerationProfile:
    """
    Model-specific defaults kept out of the public API.

    `experimental=True` means we have not yet accepted that profile for the
    initial 24 GB L4 deployment. It must be explicitly enabled by the provider.
    """

    name: str
    width: int
    height: int

    guidance_scale: float = 1.0
    pyramid_num_inference_steps_list: tuple[int, ...] = (2, 2, 2)
    amplify_first_chunk: bool = True
    enable_stage2: bool = True

    history_sizes: tuple[int, ...] = (16, 2, 1)
    num_latent_frames_per_chunk: int = 9
    keep_first_frame: bool = True

    negative_prompt: str | None = (
        "blurry, low quality, distorted anatomy, duplicated limbs, "
        "deformed faces, deformed hands, text, subtitles, watermark, "
        "compression artifacts, static frame"
    )

    experimental: bool = False


# The current Helios reference UI uses 640x384. Our existing generic API uses
# the label "480p", so for now that label maps to the conservative Helios
# standard profile rather than claiming literal 854x480 output.
HELIOS_PROFILES: dict[str, HeliosGenerationProfile] = {
    "480p": HeliosGenerationProfile(
        name="helios_distilled_standard_640x384",
        width=640,
        height=384,
    ),
    "720p": HeliosGenerationProfile(
        name="helios_distilled_720p_experimental",
        width=1280,
        height=720,
        experimental=True,
    ),
}


def get_helios_profile(
    resolution: str,
    *,
    allow_experimental: bool = False,
) -> HeliosGenerationProfile:
    try:
        profile = HELIOS_PROFILES[resolution]
    except KeyError as exc:
        supported = ", ".join(sorted(HELIOS_PROFILES))
        raise ValueError(
            f"Unsupported Helios resolution profile: {resolution!r}. "
            f"Supported values: {supported}"
        ) from exc

    if profile.experimental and not allow_experimental:
        raise ValueError(
            f"Helios profile {resolution!r} is experimental and is disabled "
            "for the initial GPU deployment."
        )

    return profile
