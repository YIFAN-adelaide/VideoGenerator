from __future__ import annotations

import re
import shutil
from pathlib import Path

from app.director.video_plan import ShotPlan
from app.providers.base import VideoProvider
from app.schemas import VideoGenerationRequest
from app.services.generated_shot import GeneratedShot
from app.services.image_probe import ImageProbe
from app.services.video_probe import VideoProbe


class FastVideoShotGeneratorError(RuntimeError):
    """Raised when one planned shot cannot be materialized."""


class FastVideoShotGenerator:
    """
    Adapter between LongVideoWorkflow's ShotGenerator protocol and the
    existing VideoProvider contract.

    Image-conditioned generation is optional:

        initial_image=None
            -> text-to-video shot

        initial_image=<previous final frame>
            -> image-conditioned next shot

    The reference image is probed here so the generated video canvas follows
    the reference image dimensions. The image should already be a model-ready
    reference stored under the shared reference_assets directory.
    """

    def __init__(
        self,
        provider: VideoProvider,
        *,
        preserve_provider_output: bool = True,
        video_probe: VideoProbe | None = None,
        image_probe: ImageProbe | None = None,
    ) -> None:
        self._provider = provider
        self._preserve_provider_output = preserve_provider_output
        self._video_probe = video_probe or VideoProbe()
        self._image_probe = image_probe or ImageProbe()

    async def generate_shot(
        self,
        *,
        shot: ShotPlan,
        prompt: str,
        output_path: Path,
        initial_image: Path | None = None,
    ) -> GeneratedShot:
        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise FastVideoShotGeneratorError(
                f"Shot {shot.shot_id} has an empty generation prompt."
            )

        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        seed = self._extract_seed(shot)

        request_kwargs: dict[str, object] = {
            "prompt": cleaned_prompt,
            "duration_seconds": shot.duration_seconds,
            "fps": shot.fps,
            "resolution": shot.resolution,
            "seed": seed,
        }

        if initial_image is not None:
            reference = (
                Path(initial_image)
                .expanduser()
                .resolve()
            )

            if not reference.exists():
                raise FastVideoShotGeneratorError(
                    f"Initial image does not exist for shot "
                    f"{shot.shot_id}: {reference}"
                )

            if not reference.is_file():
                raise FastVideoShotGeneratorError(
                    f"Initial image is not a file for shot "
                    f"{shot.shot_id}: {reference}"
                )

            if reference.stat().st_size <= 0:
                raise FastVideoShotGeneratorError(
                    f"Initial image is empty for shot "
                    f"{shot.shot_id}: {reference}"
                )

            try:
                image_info = await self._image_probe.probe(
                    reference
                )
            except Exception as exc:
                raise FastVideoShotGeneratorError(
                    f"Could not inspect initial image for shot "
                    f"{shot.shot_id}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            request_kwargs.update(
                {
                    "initial_image": str(reference),
                    "width": image_info.width,
                    "height": image_info.height,
                }
            )

        request = VideoGenerationRequest(
            **request_kwargs
        )

        provider_job_id = self._build_provider_job_id(
            shot=shot,
            output_path=target,
        )

        try:
            result = await self._provider.generate(
                request,
                provider_job_id,
            )
        except Exception as exc:
            raise FastVideoShotGeneratorError(
                f"Provider generation failed for shot "
                f"{shot.shot_id}: {type(exc).__name__}: {exc}"
            ) from exc

        provider_output = Path(result.output_path).resolve()

        if not provider_output.exists():
            raise FastVideoShotGeneratorError(
                "Provider reported success but its output file "
                f"does not exist: {provider_output}"
            )

        if not provider_output.is_file():
            raise FastVideoShotGeneratorError(
                "Provider output is not a file: "
                f"{provider_output}"
            )

        if provider_output.stat().st_size <= 0:
            raise FastVideoShotGeneratorError(
                "Provider produced an empty video file: "
                f"{provider_output}"
            )

        if provider_output != target:
            try:
                if self._preserve_provider_output:
                    shutil.copy2(
                        provider_output,
                        target,
                    )
                else:
                    shutil.move(
                        str(provider_output),
                        str(target),
                    )
            except OSError as exc:
                raise FastVideoShotGeneratorError(
                    "Could not materialize provider output at "
                    f"{target}: {exc}"
                ) from exc

        if not target.exists() or target.stat().st_size <= 0:
            raise FastVideoShotGeneratorError(
                f"Shot output was not created correctly: {target}"
            )

        try:
            probe_result = await self._video_probe.probe(target)
        except Exception as exc:
            raise FastVideoShotGeneratorError(
                f"Could not inspect generated shot "
                f"{shot.shot_id}: {type(exc).__name__}: {exc}"
            ) from exc

        return GeneratedShot(
            path=probe_result.path,
            requested_duration_seconds=float(
                shot.duration_seconds
            ),
            actual_duration_seconds=(
                probe_result.duration_seconds
            ),
            fps=probe_result.fps,
            frame_count=probe_result.frame_count,
            width=probe_result.width,
            height=probe_result.height,
        )

    @staticmethod
    def _extract_seed(
        shot: ShotPlan,
    ) -> int | None:
        raw_seed = shot.metadata.get("seed")

        if raw_seed is None:
            return None

        if isinstance(raw_seed, bool):
            raise FastVideoShotGeneratorError(
                f"Shot {shot.shot_id} metadata seed must be an integer."
            )

        try:
            return int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise FastVideoShotGeneratorError(
                f"Shot {shot.shot_id} metadata seed must be an integer."
            ) from exc

    @staticmethod
    def _build_provider_job_id(
        *,
        shot: ShotPlan,
        output_path: Path,
    ) -> str:
        long_job_id = (
            output_path.parent.parent.name
            if output_path.parent.name == "shots"
            else output_path.parent.name
        )

        raw = f"{long_job_id}_{shot.shot_id}"

        cleaned = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            raw,
        ).strip("_")

        if not cleaned:
            cleaned = f"shot_{shot.shot_id}"

        return cleaned[:180]


__all__ = [
    "FastVideoShotGenerator",
    "FastVideoShotGeneratorError",
]
