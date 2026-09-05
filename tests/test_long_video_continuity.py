from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from app.director.mock_director import MockDirector
from app.director.video_plan import DirectorRequest, ShotPlan
from app.graph.long_video_workflow import LongVideoWorkflow
from app.services.generated_shot import GeneratedShot


class ContinuityShotGenerator:
    def __init__(self) -> None:
        self.initial_images: list[Path | None] = []
        self.shot_ids: list[str] = []

    async def generate_shot(
        self,
        *,
        shot: ShotPlan,
        prompt: str,
        output_path: Path,
        initial_image: Path | None = None,
    ) -> GeneratedShot:
        self.initial_images.append(initial_image)
        self.shot_ids.append(shot.shot_id)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_bytes(
            f"fake-{shot.shot_id}".encode()
        )

        return GeneratedShot(
            path=output_path.resolve(),
            requested_duration_seconds=float(
                shot.duration_seconds
            ),
            actual_duration_seconds=5.041667,
            fps=24.0,
            frame_count=121,
            width=1280,
            height=704,
        )


class FakeFrameExtractionResult:
    def __init__(
        self,
        output_path: Path,
    ) -> None:
        self.output_path = output_path


class FakeFrameExtractor:
    def __init__(self) -> None:
        self.calls: list[
            tuple[Path, Path]
        ] = []

    async def extract_last_frame(
        self,
        video_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = True,
    ):
        source = Path(video_path).resolve()
        target = Path(output_path).resolve()

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        target.write_bytes(
            b"fake-last-frame"
        )

        self.calls.append(
            (source, target)
        )

        return FakeFrameExtractionResult(
            target
        )


@dataclass
class FakeCompositionResult:
    output_path: Path


class FakeComposer:
    async def concatenate(
        self,
        shot_paths,
        output_path,
    ):
        output = Path(output_path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_bytes(
            b"fake-final"
        )

        return FakeCompositionResult(
            output_path=output
        )


def test_workflow_chains_previous_last_frame_into_next_shot(
    tmp_path: Path,
) -> None:
    generator = ContinuityShotGenerator()
    extractor = FakeFrameExtractor()

    reference_root = (
        tmp_path / "reference_assets"
    )

    workflow = LongVideoWorkflow(
        director=MockDirector(),
        shot_generator=generator,
        composer=FakeComposer(),
        output_dir=tmp_path / "outputs",
        frame_extractor=extractor,
        reference_asset_dir=reference_root,
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
            job_id="job-continuity",
        )
    )

    assert result["status"] == "completed"

    assert generator.shot_ids == [
        "shot_001",
        "shot_002",
        "shot_003",
    ]

    shot_1_ref = (
        reference_root
        / "jobs"
        / "job-continuity"
        / "shot_001_last.png"
    ).resolve()

    shot_2_ref = (
        reference_root
        / "jobs"
        / "job-continuity"
        / "shot_002_last.png"
    ).resolve()

    shot_3_ref = (
        reference_root
        / "jobs"
        / "job-continuity"
        / "shot_003_last.png"
    ).resolve()

    # Shot 1 starts from text only.
    assert generator.initial_images[0] is None

    # Shot 2 starts from the exact final frame of Shot 1.
    assert (
        generator.initial_images[1]
        == shot_1_ref
    )

    # Shot 3 starts from the exact final frame of Shot 2.
    assert (
        generator.initial_images[2]
        == shot_2_ref
    )

    assert result[
        "continuity_previous_last_frame"
    ] == str(shot_3_ref)

    assert result[
        "continuity_reference_history"
    ] == [
        str(shot_1_ref),
        str(shot_2_ref),
        str(shot_3_ref),
    ]

    completed = result["completed_shots"]

    assert completed[0][
        "initial_image_path"
    ] is None

    assert completed[0][
        "last_frame_reference_path"
    ] == str(shot_1_ref)

    assert completed[1][
        "initial_image_path"
    ] == str(shot_1_ref)

    assert completed[1][
        "last_frame_reference_path"
    ] == str(shot_2_ref)

    assert completed[2][
        "initial_image_path"
    ] == str(shot_2_ref)

    assert completed[2][
        "last_frame_reference_path"
    ] == str(shot_3_ref)


def test_workflow_can_start_first_shot_from_existing_reference(
    tmp_path: Path,
) -> None:
    generator = ContinuityShotGenerator()
    extractor = FakeFrameExtractor()

    initial = (
        tmp_path
        / "reference_assets"
        / "prepared"
        / "user.png"
    )
    initial.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    initial.write_bytes(
        b"user-reference"
    )

    workflow = LongVideoWorkflow(
        director=MockDirector(),
        shot_generator=generator,
        composer=FakeComposer(),
        output_dir=tmp_path / "outputs",
        frame_extractor=extractor,
        reference_asset_dir=(
            tmp_path / "reference_assets"
        ),
    )

    result = asyncio.run(
        workflow.run(
            DirectorRequest(
                prompt="A tiger story.",
                target_duration_seconds=5,
                max_shot_duration_seconds=5,
            ),
            job_id="job-user-image",
            initial_image=initial,
        )
    )

    assert result["status"] == "completed"
    assert generator.initial_images[0] == (
        initial.resolve()
    )
