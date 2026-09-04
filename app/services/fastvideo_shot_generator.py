from __future__ import annotations

import re
import shutil
from pathlib import Path

from app.director.video_plan import ShotPlan
from app.providers.base import VideoProvider
from app.schemas import VideoGenerationRequest


class FastVideoShotGeneratorError(RuntimeError):
    """Raised when one planned shot cannot be materialized."""


class FastVideoShotGenerator:
    """
    Adapter between LongVideoWorkflow's ShotGenerator protocol and the
    existing VideoProvider contract.

    The existing provider API remains unchanged:

        await provider.generate(
            request: VideoGenerationRequest,
            job_id: str,
        )

    This adapter translates one ShotPlan into one VideoGenerationRequest,
    invokes the provider, and materializes the returned MP4 at the path
    requested by LongVideoWorkflow.
    """

    def __init__(
        self,
        provider: VideoProvider,
        *,
        preserve_provider_output: bool = True,
    ) -> None:
        self._provider = provider
        self._preserve_provider_output = preserve_provider_output

    async def generate_shot(
        self,
        *,
        shot: ShotPlan,
        prompt: str,
        output_path: Path,
    ) -> Path:
        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise FastVideoShotGeneratorError(
                f"Shot {shot.shot_id} has an empty generation prompt."
            )

        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        seed = self._extract_seed(shot)

        request = VideoGenerationRequest(
            prompt=cleaned_prompt,
            duration_seconds=shot.duration_seconds,
            fps=shot.fps,
            resolution=shot.resolution,
            seed=seed,
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

        if provider_output == target:
            return target

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

        return target

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
        """
        Create a traceable provider job id from the long-video job directory
        and shot id.

        Expected workflow path:
            <output_dir>/<long_job_id>/shots/<shot_id>.mp4

        which becomes:
            <long_job_id>_<shot_id>
        """
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
