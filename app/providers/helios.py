from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path
from typing import Any, Callable

from app.providers.base import ProviderResult, VideoProvider
from app.providers.helios_parameters import (
    HeliosGenerationParams,
    normalize_helios_frame_count,
    resolve_seed,
)
from app.providers.helios_profiles import get_helios_profile
from app.runtime.helios_loader import HeliosModelLoader
from app.schemas import VideoGenerationRequest


class HeliosGenerationError(RuntimeError):
    """Raised when a Helios inference or export operation fails."""


class HeliosProvider(VideoProvider):
    """
    Adapter between our generic VideoGenerationRequest and Helios.

    Responsibilities:
    1. Convert generic API parameters into HeliosGenerationParams.
    2. Reuse the pipeline owned by HeliosModelLoader.
    3. Run blocking GPU inference outside the asyncio event loop.
    4. Export frames to MP4.
    5. Return a generic ProviderResult.

    It does NOT load model weights itself.
    """

    def __init__(
        self,
        loader: HeliosModelLoader,
        output_dir: str = "./outputs",
        *,
        allow_experimental_profiles: bool = False,
        exporter: Callable[..., Any] | None = None,
        generator_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self.loader = loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.allow_experimental_profiles = allow_experimental_profiles

        # Injection points keep local unit tests independent of CUDA/diffusers.
        self._exporter_override = exporter
        self._generator_factory_override = generator_factory

    async def generate(
        self,
        request: VideoGenerationRequest,
        job_id: str,
    ) -> ProviderResult:
        params = self.build_params(request)

        # Helios inference and video encoding are synchronous/blocking work.
        # Running them in a worker thread prevents the FastAPI asyncio loop
        # from being blocked for the entire generation.
        return await asyncio.to_thread(
            self._generate_sync,
            params,
            job_id,
        )

    def build_params(
        self,
        request: VideoGenerationRequest,
    ) -> HeliosGenerationParams:
        profile = get_helios_profile(
            request.resolution,
            allow_experimental=self.allow_experimental_profiles,
        )

        (
            requested_frames,
            num_frames,
            actual_duration_seconds,
        ) = normalize_helios_frame_count(
            request.duration_seconds,
            request.fps,
        )

        seed = resolve_seed(request.seed)

        return HeliosGenerationParams(
            prompt=request.prompt.strip(),
            negative_prompt=profile.negative_prompt,
            profile_name=profile.name,
            requested_resolution=request.resolution,
            width=profile.width,
            height=profile.height,
            requested_duration_seconds=request.duration_seconds,
            fps=request.fps,
            requested_frames=requested_frames,
            num_frames=num_frames,
            actual_duration_seconds=actual_duration_seconds,
            seed=seed,
            guidance_scale=profile.guidance_scale,
            pyramid_num_inference_steps_list=(
                profile.pyramid_num_inference_steps_list
            ),
            amplify_first_chunk=profile.amplify_first_chunk,
            enable_stage2=profile.enable_stage2,
            history_sizes=profile.history_sizes,
            num_latent_frames_per_chunk=(
                profile.num_latent_frames_per_chunk
            ),
            keep_first_frame=profile.keep_first_frame,
        )

    def _generate_sync(
        self,
        params: HeliosGenerationParams,
        job_id: str,
    ) -> ProviderResult:
        pipeline = self.loader.get_pipeline()

        output_path = self.output_dir / f"{job_id}.mp4"
        generator = self._create_generator(params.seed)

        pipeline_kwargs = self._build_pipeline_kwargs(
            pipeline=pipeline,
            params=params,
            generator=generator,
        )

        started = time.perf_counter()

        try:
            output = pipeline(**pipeline_kwargs)
            frames = output.frames[0]
        except Exception as exc:
            raise HeliosGenerationError(
                f"Helios inference failed for job {job_id}: {exc}"
            ) from exc

        inference_seconds = time.perf_counter() - started

        try:
            exporter = self._get_exporter()
            exporter(
                frames,
                str(output_path),
                fps=params.fps,
            )
        except Exception as exc:
            # Avoid returning a partial/corrupt output as if generation
            # succeeded.
            output_path.unlink(missing_ok=True)
            raise HeliosGenerationError(
                f"Video export failed for job {job_id}: {exc}"
            ) from exc

        metadata = {
            "provider": "helios",
            "profile": params.profile_name,
            "requested_resolution": params.requested_resolution,
            "width": params.width,
            "height": params.height,
            "fps": params.fps,
            "requested_duration_seconds": (
                params.requested_duration_seconds
            ),
            "actual_duration_seconds": params.actual_duration_seconds,
            "requested_frames": params.requested_frames,
            "num_frames": params.num_frames,
            "seed": params.seed,
            "guidance_scale": params.guidance_scale,
            "pyramid_num_inference_steps_list": list(
                params.pyramid_num_inference_steps_list
            ),
            "inference_seconds": inference_seconds,
        }

        return ProviderResult(
            output_path=str(output_path.resolve()),
            metadata=metadata,
        )

    def _build_pipeline_kwargs(
        self,
        *,
        pipeline: Any,
        params: HeliosGenerationParams,
        generator: Any,
    ) -> dict[str, Any]:
        """
        Build kwargs compatible with both:
        - repository-native HeliosPipeline, and
        - newer Diffusers HeliosPyramidPipeline.

        The repository-native pipeline has a few explicit stage/history
        parameters that the newer Pyramid pipeline internalizes. We only
        forward optional compatibility parameters when supported.
        """
        kwargs: dict[str, Any] = {
            "prompt": params.prompt,
            "height": params.height,
            "width": params.width,
            "num_frames": params.num_frames,
            "guidance_scale": params.guidance_scale,
            "generator": generator,
            "output_type": params.output_type,
            "pyramid_num_inference_steps_list": list(
                params.pyramid_num_inference_steps_list
            ),
            "is_amplify_first_chunk": params.amplify_first_chunk,
        }

        if params.negative_prompt:
            kwargs["negative_prompt"] = params.negative_prompt

        optional_native_kwargs = {
            "is_enable_stage2": params.enable_stage2,
            "history_sizes": list(params.history_sizes),
            "num_latent_frames_per_chunk": (
                params.num_latent_frames_per_chunk
            ),
            "keep_first_frame": params.keep_first_frame,
        }

        signature = inspect.signature(pipeline.__call__)
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        if accepts_var_kwargs:
            kwargs.update(optional_native_kwargs)
        else:
            for key, value in optional_native_kwargs.items():
                if key in signature.parameters:
                    kwargs[key] = value

        return kwargs

    def _create_generator(self, seed: int) -> Any:
        if self._generator_factory_override is not None:
            return self._generator_factory_override(seed)

        try:
            import torch
        except ImportError as exc:
            raise HeliosGenerationError(
                "PyTorch is unavailable in the Helios runtime."
            ) from exc

        device = getattr(
            getattr(self.loader, "config", None),
            "device",
            "cuda:0",
        )
        return torch.Generator(device=device).manual_seed(seed)

    def _get_exporter(self) -> Callable[..., Any]:
        if self._exporter_override is not None:
            return self._exporter_override

        try:
            from diffusers.utils import export_to_video
        except ImportError as exc:
            raise HeliosGenerationError(
                "diffusers.utils.export_to_video is unavailable."
            ) from exc

        return export_to_video
