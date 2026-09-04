from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.director.mock_director import MockDirector
from app.director.video_plan import DirectorRequest, ShotPlan
from app.graph.long_video_workflow import LongVideoWorkflow
from app.services.generated_shot import GeneratedShot


class FakeShotGenerator:
    def __init__(self) -> None:
        self.received_prompts: list[str] = []
        self.received_shots: list[str] = []

    async def generate_shot(
        self,
        *,
        shot: ShotPlan,
        prompt: str,
        output_path: Path,
    ) -> Path:
        self.received_prompts.append(prompt)
        self.received_shots.append(shot.shot_id)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_bytes(
            f"fake-{shot.shot_id}".encode("utf-8")
        )

        return output_path


class MetadataShotGenerator(FakeShotGenerator):
    async def generate_shot(
        self,
        *,
        shot: ShotPlan,
        prompt: str,
        output_path: Path,
    ) -> GeneratedShot:
        path = await super().generate_shot(
            shot=shot,
            prompt=prompt,
            output_path=output_path,
        )

        return GeneratedShot(
            path=path.resolve(),
            requested_duration_seconds=float(
                shot.duration_seconds
            ),
            actual_duration_seconds=4.875,
            fps=24.0,
            frame_count=117,
            width=1280,
            height=704,
        )


@dataclass
class FakeCompositionResult:
    output_path: Path


class FakeComposer:
    def __init__(self) -> None:
        self.received_inputs: list[str] = []

    async def concatenate(
        self,
        shot_paths,
        output_path,
    ):
        self.received_inputs = [
            str(path)
            for path in shot_paths
        ]

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_bytes(
            b"fake-final-video"
        )

        return FakeCompositionResult(
            output_path=output
        )


def test_long_video_workflow_generates_three_shots_and_composes(
    tmp_path: Path,
):
    generator = FakeShotGenerator()
    composer = FakeComposer()

    workflow = LongVideoWorkflow(
        director=MockDirector(),
        shot_generator=generator,
        composer=composer,
        output_dir=tmp_path,
        video_prompt_language="en",
    )

    result = asyncio.run(
        workflow.run(
            DirectorRequest(
                prompt=(
                    "A tiger walks through a snowy forest."
                ),
                target_duration_seconds=15,
                max_shot_duration_seconds=5,
            ),
            job_id="job-001",
        )
    )

    assert result["status"] == "completed"
    assert result["error"] is None

    assert generator.received_shots == [
        "shot_001",
        "shot_002",
        "shot_003",
    ]

    assert (
        result["current_shot_index"]
        == 3
    )

    assert len(
        result["completed_shot_paths"]
    ) == 3

    assert len(
        result["completed_shots"]
    ) == 3

    # Legacy Path-returning generators remain supported.
    assert all(
        item["actual_duration_seconds"] is None
        for item in result["completed_shots"]
    )

    assert len(
        composer.received_inputs
    ) == 3

    final_path = Path(
        result["final_output_path"]
    )

    assert final_path.exists()
    assert final_path.name == "final.mp4"


def test_long_video_workflow_records_real_shot_metadata(
    tmp_path: Path,
):
    workflow = LongVideoWorkflow(
        director=MockDirector(),
        shot_generator=MetadataShotGenerator(),
        composer=FakeComposer(),
        output_dir=tmp_path,
        video_prompt_language="en",
    )

    result = asyncio.run(
        workflow.run(
            DirectorRequest(
                prompt="A cinematic tiger story.",
                target_duration_seconds=15,
                max_shot_duration_seconds=5,
            ),
            job_id="job-metadata",
        )
    )

    assert result["status"] == "completed"

    completed = result["completed_shots"]
    assert len(completed) == 3

    first = completed[0]

    assert first["shot_id"] == "shot_001"
    assert first["requested_duration_seconds"] == 5.0
    assert first["actual_duration_seconds"] == pytest.approx(
        4.875
    )
    assert first["duration_delta_seconds"] == pytest.approx(
        -0.125
    )
    assert first["actual_fps"] == pytest.approx(24.0)
    assert first["frame_count"] == 117
    assert first["width"] == 1280
    assert first["height"] == 704

    assert sum(
        item["actual_duration_seconds"]
        for item in completed
        if item["actual_duration_seconds"] is not None
    ) == pytest.approx(14.625)


def test_long_video_workflow_handles_partial_final_shot(
    tmp_path: Path,
):
    workflow = LongVideoWorkflow(
        director=MockDirector(),
        shot_generator=FakeShotGenerator(),
        composer=FakeComposer(),
        output_dir=tmp_path,
    )

    result = asyncio.run(
        workflow.run(
            DirectorRequest(
                prompt="A cinematic tiger story.",
                target_duration_seconds=12,
                max_shot_duration_seconds=5,
            ),
            job_id="job-002",
        )
    )

    assert result["status"] == "completed"

    plan = result["video_plan"]

    assert [
        shot.duration_seconds
        for shot in plan.shots
    ] == [5, 5, 2]


def test_long_video_workflow_uses_chinese_generation_prompts(
    tmp_path: Path,
):
    generator = FakeShotGenerator()

    workflow = LongVideoWorkflow(
        director=MockDirector(),
        shot_generator=generator,
        composer=FakeComposer(),
        output_dir=tmp_path,
        video_prompt_language="zh",
    )

    result = asyncio.run(
        workflow.run(
            DirectorRequest(
                prompt="一只小老虎走过雪地森林。",
                target_duration_seconds=10,
            ),
            job_id="job-003",
        )
    )

    assert result["status"] == "completed"

    assert len(
        generator.received_prompts
    ) == 2

    assert all(
        "电影感镜头" in prompt
        for prompt in generator.received_prompts
    )


def test_long_video_workflow_stops_when_shot_generation_fails(
    tmp_path: Path,
):
    class FailingShotGenerator:
        async def generate_shot(
            self,
            *,
            shot,
            prompt,
            output_path,
        ):
            raise RuntimeError(
                "simulated generation failure"
            )

    composer = FakeComposer()

    workflow = LongVideoWorkflow(
        director=MockDirector(),
        shot_generator=FailingShotGenerator(),
        composer=composer,
        output_dir=tmp_path,
    )

    result = asyncio.run(
        workflow.run(
            DirectorRequest(
                prompt="A tiger story.",
                target_duration_seconds=10,
            ),
            job_id="job-004",
        )
    )

    assert result["status"] == "failed"
    assert "simulated generation failure" in (
        result["error"]
    )

    assert composer.received_inputs == []


def test_long_video_workflow_rejects_invalid_prompt_language(
    tmp_path: Path,
):
    with pytest.raises(
        ValueError,
        match="must be 'en' or 'zh'",
    ):
        LongVideoWorkflow(
            director=MockDirector(),
            shot_generator=FakeShotGenerator(),
            composer=FakeComposer(),
            output_dir=tmp_path,
            video_prompt_language="fr",
        )
