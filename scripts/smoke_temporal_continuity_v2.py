from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from app.config import Settings
from app.director.mock_director import MockDirector
from app.director.video_plan import DirectorRequest
from app.graph.long_video_workflow import LongVideoWorkflow
from app.providers.factory import build_provider_resources
from app.services.fastvideo_shot_generator import (
    FastVideoShotGenerator,
)
from app.services.frame_extractor import FrameExtractor
from app.services.temporal_continuity import (
    TemporalContinuityState,
)
from app.services.temporal_continuity_provider import (
    StaticTemporalContinuityProvider,
)
from app.services.video_composer import VideoComposer


def _build_test_continuity() -> StaticTemporalContinuityProvider:
    """
    Fixed V2 experiment.

    This is intentionally deterministic so the V1-vs-V2 comparison changes
    only the semantic continuity instructions. It is NOT the final product
    path; the future vLLM Director will produce these states automatically.
    """
    return StaticTemporalContinuityProvider(
        {
            "shot_002": TemporalContinuityState(
                ongoing_process=(
                    "the same tiger is already walking steadily "
                    "through the snowy forest"
                ),
                current_phase=(
                    "the tiger is mid-walk and must continue from "
                    "its current pose and screen position"
                ),
                next_development=(
                    "continue the same walking action naturally "
                    "without stopping or restarting"
                ),
                spatial_change=(
                    "continue in the same established travel "
                    "direction at approximately the same pace"
                ),
                camera_behavior=(
                    "continue tracking the tiger smoothly in the "
                    "same direction"
                ),
                camera_framing=(
                    "maintain approximately the same camera distance "
                    "and composition"
                ),
                preserve=(
                    "the same tiger identity, proportions and fur pattern",
                    "the same snowy forest, lighting and visual style",
                    "the tiger's current screen position as closely as possible",
                ),
                avoid=(
                    "repositioning the tiger to a new starting point",
                    "turning the tiger around without narrative reason",
                    "resetting the walking action",
                    "sudden camera reframing",
                ),
            ),
            "shot_003": TemporalContinuityState(
                ongoing_process=(
                    "the same tiger is continuing the same walk "
                    "through the same snowy forest"
                ),
                current_phase=(
                    "the walking action is already in progress"
                ),
                next_development=(
                    "continue naturally from the previous moment "
                    "with no action reset"
                ),
                spatial_change=(
                    "maintain the established travel direction and "
                    "approximately the same walking pace"
                ),
                camera_behavior=(
                    "continue the same smooth tracking behavior"
                ),
                camera_framing=(
                    "preserve approximately the same distance and "
                    "relative subject framing"
                ),
                preserve=(
                    "the same tiger identity, proportions and fur pattern",
                    "the same snowy forest, lighting and visual style",
                    "the current visual composition as closely as possible",
                ),
                avoid=(
                    "starting a new independent walking shot",
                    "jumping the tiger to a new screen position",
                    "reversing movement direction",
                    "sudden camera reset",
                ),
            ),
        }
    )


async def main() -> None:
    settings = Settings()
    resources = build_provider_resources(
        settings
    )

    job_id = (
        "continuity-v2-smoke-"
        + uuid4().hex[:8]
    )

    workflow = LongVideoWorkflow(
        director=MockDirector(),
        shot_generator=FastVideoShotGenerator(
            resources.provider
        ),
        composer=VideoComposer(),
        output_dir=settings.output_dir,
        video_prompt_language="en",
        frame_extractor=FrameExtractor(),
        reference_asset_dir=(
            Path("reference_assets").resolve()
        ),
        temporal_continuity_provider=(
            _build_test_continuity()
        ),
    )

    print("=" * 72)
    print("TEMPORAL CONTINUITY V2 SMOKE TEST")
    print("=" * 72)
    print("job:", job_id)
    print()
    print(
        "V2 adds semantic temporal instructions to Shot 2 and Shot 3 "
        "while keeping the same previous-last-frame conditioning."
    )
    print()

    result = await workflow.run(
        DirectorRequest(
            prompt=(
                "A cinematic tiger walks continuously through "
                "a snowy forest while the camera follows it."
            ),
            target_duration_seconds=15,
            max_shot_duration_seconds=5,
            fps=24,
            resolution="720p",
        ),
        job_id=job_id,
    )

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print("status:", result["status"])
    print("error:", result["error"])
    print("final:", result["final_output_path"])

    for shot in result["completed_shots"]:
        print()
        print("-" * 72)
        print("shot:", shot["shot_id"])
        print("initial image:", shot["initial_image_path"])
        print("video:", shot["path"])
        print(
            "temporal state:",
            shot["temporal_continuity"],
        )
        print()
        print("EFFECTIVE FASTWAN PROMPT")
        print(shot["effective_generation_prompt"])

    close = getattr(
        resources.provider,
        "close",
        None,
    )
    if close is not None:
        await close()


if __name__ == "__main__":
    asyncio.run(main())
