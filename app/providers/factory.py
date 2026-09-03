from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.providers.base import VideoProvider
from app.providers.fastvideo import FastVideoProvider
from app.providers.helios import HeliosProvider
from app.providers.mock import MockVideoProvider
from app.runtime.helios_loader import HeliosModelLoader, HeliosRuntimeConfig


@dataclass(frozen=True)
class ProviderResources:
    """
    Everything created for one provider selection.

    `provider` is always present.
    `helios_loader` is present only when VIDEO_PROVIDER=helios because Helios
    is still an in-process legacy/reference runtime. External runtimes such as
    FastVideo own their model lifecycle in a separate serving process.
    """

    provider: VideoProvider
    helios_loader: HeliosModelLoader | None = None


def build_provider_resources(settings: Settings) -> ProviderResources:
    if settings.video_provider == "mock":
        return ProviderResources(
            provider=MockVideoProvider(settings.output_dir)
        )

    if settings.video_provider == "fastvideo":
        return ProviderResources(
            provider=FastVideoProvider(
                base_url=settings.fastvideo_base_url,
                model=settings.fastvideo_model,
                output_dir=settings.output_dir,
                poll_interval_seconds=(
                    settings.fastvideo_poll_interval_seconds
                ),
                request_timeout_seconds=(
                    settings.fastvideo_request_timeout_seconds
                ),
                job_timeout_seconds=settings.fastvideo_job_timeout_seconds,
                size_480p=settings.fastvideo_480p_size,
                size_720p=settings.fastvideo_720p_size,
            )
        )

    if settings.video_provider == "helios":
        runtime_config = HeliosRuntimeConfig(
            base_model_path=settings.helios_base_model_path,
            transformer_path=settings.helios_transformer_path,
            repo_path=settings.helios_repo_path,
            device=settings.helios_device,
            weight_dtype=settings.helios_weight_dtype,
            low_vram=settings.helios_low_vram,
            group_offloading_type=settings.helios_group_offloading_type,
            num_blocks_per_group=settings.helios_num_blocks_per_group,
            enable_compile=settings.helios_enable_compile,
            disable_flash_attention=(
                settings.helios_disable_flash_attention
            ),
        )

        loader = HeliosModelLoader(runtime_config)

        provider = HeliosProvider(
            loader=loader,
            output_dir=settings.output_dir,
            allow_experimental_profiles=False,
        )

        return ProviderResources(
            provider=provider,
            helios_loader=loader,
        )

    raise ValueError(
        f"Unsupported VIDEO_PROVIDER: {settings.video_provider!r}"
    )


def build_provider(settings: Settings) -> VideoProvider:
    """
    Backwards-compatible helper for code that only needs the provider.

    New application bootstrap code should prefer build_provider_resources()
    because it also exposes lifecycle-managed resources.
    """
    return build_provider_resources(settings).provider
