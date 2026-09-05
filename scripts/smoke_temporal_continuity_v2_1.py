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


def _build_v2_1_continuity() -> StaticTemporalContinuityProvider:
    """
    Deliberately fixed hypothesis test:
    if the tiger reaches the right side, keep its world-space progression
    but make the camera follow/reveal more environment instead of resetting
    the tiger to the left.
    """
    common_preserve = (
        "the same tiger identity, proportions and fur pattern",
        "the same snowy forest, lighting and visual style",
        "the ongoing walking action",
    )

    return StaticTemporalContinuityProvider(
        {
            "shot_002": TemporalContinuityState(
                ongoing_process=(
                    "the same tiger is already walking steadily "
                    "through the snowy forest"
                ),
                current_phase=(
                    "the tiger is already well into the walk and may be "
                    "near the right side of the supplied reference frame"
                ),
                next_development=(
                    "continue the existing walking action naturally "
                    "without restarting"
                ),
                spatial_change=(
                    "continue forward through the forest in the same "
                    "established world-space direction and pace"
                ),
                subject_screen_behavior=(
                    "do not move or reset the tiger back to the left side; "
                    "keep it in a natural tracking region of the frame"
                ),
                camera_response=(
                    "smoothly pan and track in the tiger's travel direction "
                    "so the camera follows the ongoing movement"
                ),
                environment_reveal=(
                    "reveal new snowy forest ahead in the tiger's travel "
                    "direction as the camera follows"
                ),
                camera_behavior=(
                    "continue one smooth tracking movement rather than "
                    "starting a newly composed camera shot"
                ),
                camera_framing=(
                    "preserve approximately the current subject scale and "
                    "camera distance"
                ),
                preserve=common_preserve,
                avoid=(
                    "repositioning the tiger to a new starting point",
                    "making the tiger cross the entire frame from left to right again",
                    "resetting the walking action",
                    "changing to a new wide, medium, or close composition abruptly",
                ),
            ),
            "shot_003": TemporalContinuityState(
                ongoing_process=(
                    "the same tiger continues the uninterrupted walk"
                ),
                current_phase=(
                    "the tiger is already moving and the camera is already "
                    "following it"
                ),
                next_development=(
                    "continue the same movement without a new start"
                ),
                spatial_change=(
                    "continue deeper through the forest in the already "
                    "established world-space direction"
                ),
                subject_screen_behavior=(
                    "keep the tiger naturally framed from its current screen "
                    "position instead of resetting it to the opposite side"
                ),
                camera_response=(
                    "continue the existing tracking/panning response so the "
                    "camera travels with the tiger"
                ),
                environment_reveal=(
                    "continue revealing more forest ahead rather than "
                    "replaying the previous screen traversal"
                ),
                camera_behavior=(
                    "maintain the same continuous tracking behavior"
                ),
                camera_framing=(
                    "avoid a sudden close-up or other composition reset"
                ),
                preserve=common_preserve,
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
    resources = build_provider_resources(settings)

    job_id = (
        "continuity-v2-1-smoke-"
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
            _build_v2_1_continuity()
        ),
    )

    print("=" * 72)
    print("TEMPORAL CONTINUITY V2.1 SMOKE TEST")
    print("=" * 72)
    print("job:", job_id)
    print()
    print(
        "Hypothesis: when the subject reaches the edge of the frame, "
        "the camera should follow and reveal new world space rather "
        "than resetting the subject and replaying the traversal."
    )

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
        print()
        print("ORIGINAL DIRECTOR PROMPT")
        print(shot["base_generation_prompt"])
        print()
        print("CONTINUATION BASE PROMPT")
        print(shot["continuation_base_prompt"])
        print()
        print("EFFECTIVE FASTWAN PROMPT")
        print(shot["effective_generation_prompt"])

    close = getattr(resources.provider, "close", None)
    if close is not None:
        await close()


if __name__ == "__main__":
    asyncio.run(main())
