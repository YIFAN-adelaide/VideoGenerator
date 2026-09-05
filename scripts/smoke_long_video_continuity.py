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
from app.services.video_composer import VideoComposer


async def main() -> None:
    settings = Settings()
    resources = build_provider_resources(
        settings
    )

    job_id = (
        "continuity-smoke-"
        + uuid4().hex[:8]
    )

    generator = FastVideoShotGenerator(
        resources.provider
    )

    workflow = LongVideoWorkflow(
        director=MockDirector(),
        shot_generator=generator,
        composer=VideoComposer(),
        output_dir=settings.output_dir,
        video_prompt_language="en",
        frame_extractor=FrameExtractor(),
        reference_asset_dir=(
            Path("reference_assets").resolve()
        ),
    )

    print("=" * 72)
    print("LONG VIDEO CONTINUITY SMOKE TEST")
    print("=" * 72)
    print("job:", job_id)
    print()
    print(
        "Expected conditioning flow:\n"
        "shot_001: text only\n"
        "shot_002: shot_001_last.png\n"
        "shot_003: shot_002_last.png"
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
    print(
        "final:",
        result["final_output_path"],
    )
    print(
        "last continuity frame:",
        result[
            "continuity_previous_last_frame"
        ],
    )

    print()
    print("SHOTS")

    for shot in result["completed_shots"]:
        print("-" * 72)
        print("shot:", shot["shot_id"])
        print(
            "initial image:",
            shot["initial_image_path"],
        )
        print(
            "last frame:",
            shot[
                "last_frame_reference_path"
            ],
        )
        print(
            "video:",
            shot["path"],
        )
        print(
            "frames:",
            shot["frame_count"],
        )
        print(
            "size:",
            f"{shot['width']}x{shot['height']}",
        )
        print(
            "duration:",
            shot[
                "actual_duration_seconds"
            ],
        )

    close = getattr(
        resources.provider,
        "close",
        None,
    )
    if close is not None:
        await close()


if __name__ == "__main__":
    asyncio.run(main())
