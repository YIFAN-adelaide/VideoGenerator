from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

import app.providers.factory as provider_factory

from app.director.mock_director import MockDirector
from app.director.video_plan import DirectorRequest
from app.graph.long_video_workflow import LongVideoWorkflow
from app.services.fastvideo_shot_generator import FastVideoShotGenerator
from app.services.video_composer import VideoComposer


async def main() -> None:
    settings = provider_factory.Settings()

    resources = provider_factory.build_provider_resources(
        settings
    )

    provider = resources.provider

    print("=" * 70)
    print("RUNTIME")
    print("=" * 70)
    print("provider:", type(provider).__name__)
    print("fastvideo:", settings.fastvideo_base_url)
    print("model:", settings.fastvideo_model)
    print("output_dir:", settings.output_dir)

    if type(provider).__name__ != "FastVideoProvider":
        raise RuntimeError(
            "Smoke test requires FastVideoProvider."
        )

    shot_generator = FastVideoShotGenerator(
        provider,
        preserve_provider_output=True,
    )

    workflow = LongVideoWorkflow(
        director=MockDirector(),
        shot_generator=shot_generator,
        composer=VideoComposer(),
        output_dir=settings.output_dir,
        video_prompt_language="en",
    )

    request = DirectorRequest(
        prompt=(
            "A cinematic tiger walks slowly through a snowy "
            "forest at sunrise. The same tiger remains visually "
            "consistent throughout the sequence. Soft golden "
            "sunlight, realistic fur, snowy pine forest, "
            "cinematic low-angle tracking camera."
        ),
        target_duration_seconds=15,
        max_shot_duration_seconds=5,
        fps=24,
        resolution="720p",
        user_language="en",
        planning_language="en",
    )

    job_id = (
        f"long-smoke-{uuid4().hex[:8]}"
    )

    print("\nStarting job:", job_id)

    result = await workflow.run(
        request,
        job_id=job_id,
    )

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print("status:", result["status"])
    print("error:", result["error"])
    print(
        "completed shots:",
        len(result["completed_shot_paths"]),
    )

    for index, shot_path in enumerate(
        result["completed_shot_paths"],
        start=1,
    ):
        path = Path(shot_path)

        print(
            f"shot {index}:",
            path,
            f"({path.stat().st_size} bytes)"
            if path.exists()
            else "(missing)",
        )

    print(
        "final_output:",
        result["final_output_path"],
    )

    if result["status"] != "completed":
        raise SystemExit(1)

    final_path = Path(
        result["final_output_path"]
    )

    print(
        "final_size:",
        final_path.stat().st_size,
        "bytes",
    )


if __name__ == "__main__":
    asyncio.run(main())