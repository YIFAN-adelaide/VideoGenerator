from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.director.base import BaseDirector
from app.director.video_plan import DirectorRequest, ShotPlan, VideoPlan
from app.services.frame_extractor import FrameExtractor
from app.services.generated_shot import GeneratedShot
from app.services.shot_continuity import ShotContinuityState
from app.services.video_composer import VideoComposer


class ShotGenerator(Protocol):
    async def generate_shot(
        self,
        *,
        shot: ShotPlan,
        prompt: str,
        output_path: Path,
        initial_image: Path | None = None,
    ) -> GeneratedShot | str | Path:
        ...


class CompletedShotInfo(TypedDict):
    shot_id: str
    path: str

    requested_duration_seconds: float
    actual_duration_seconds: float | None
    duration_delta_seconds: float | None

    actual_fps: float | None
    frame_count: int | None
    width: int | None
    height: int | None

    initial_image_path: str | None
    last_frame_reference_path: str | None


class LongVideoState(TypedDict):
    job_id: str
    request: DirectorRequest

    video_plan: VideoPlan | None

    current_shot_index: int
    completed_shot_paths: list[str]
    completed_shots: list[CompletedShotInfo]

    continuity_previous_last_frame: str | None
    continuity_reference_history: list[str]

    status: str
    final_output_path: str | None
    error: str | None


class LongVideoWorkflow:
    """
    Sequential long-video workflow with optional shot-to-shot continuity.

    Continuity is enabled when ``frame_extractor`` is provided:

        Shot N
          -> generate
          -> extract exact final frame
          -> store under reference_assets/jobs/<job_id>/
          -> pass that PNG as initial_image for Shot N+1

    If ``frame_extractor`` is omitted, the workflow behaves like the previous
    implementation and does not pass initial_image arguments to legacy fake
    generators. This keeps existing tests/backends backward compatible.
    """

    def __init__(
        self,
        *,
        director: BaseDirector,
        shot_generator: ShotGenerator,
        composer: VideoComposer,
        output_dir: str | Path,
        video_prompt_language: str = "en",
        frame_extractor: FrameExtractor | None = None,
        reference_asset_dir: str | Path = "reference_assets",
    ) -> None:
        if video_prompt_language not in {"en", "zh"}:
            raise ValueError(
                "video_prompt_language must be 'en' or 'zh'."
            )

        self._director = director
        self._shot_generator = shot_generator
        self._composer = composer
        self._output_dir = Path(output_dir)
        self._video_prompt_language = video_prompt_language

        self._frame_extractor = frame_extractor
        self._reference_asset_dir = (
            Path(reference_asset_dir)
            .expanduser()
            .resolve()
        )

        self.graph = self._build_graph()

    @property
    def continuity_enabled(self) -> bool:
        return self._frame_extractor is not None

    def _build_graph(self):
        builder = StateGraph(LongVideoState)

        builder.add_node(
            "plan_video",
            self._plan_video,
        )
        builder.add_node(
            "generate_current_shot",
            self._generate_current_shot,
        )
        builder.add_node(
            "compose_video",
            self._compose_video,
        )

        builder.add_edge(
            START,
            "plan_video",
        )

        builder.add_conditional_edges(
            "plan_video",
            self._route_after_plan,
            {
                "generate": "generate_current_shot",
                "failed": END,
            },
        )

        builder.add_conditional_edges(
            "generate_current_shot",
            self._route_after_generation,
            {
                "generate": "generate_current_shot",
                "compose": "compose_video",
                "failed": END,
            },
        )

        builder.add_edge(
            "compose_video",
            END,
        )

        return builder.compile()

    async def run(
        self,
        request: DirectorRequest,
        *,
        job_id: str | None = None,
        initial_image: str | Path | None = None,
    ) -> LongVideoState:
        resolved_job_id = job_id or uuid4().hex

        initial_reference: str | None = None

        if initial_image is not None:
            candidate = (
                Path(initial_image)
                .expanduser()
                .resolve()
            )

            if not candidate.exists():
                raise FileNotFoundError(
                    f"Initial continuity image does not exist: {candidate}"
                )

            if not candidate.is_file():
                raise ValueError(
                    f"Initial continuity image is not a file: {candidate}"
                )

            if candidate.stat().st_size <= 0:
                raise ValueError(
                    f"Initial continuity image is empty: {candidate}"
                )

            initial_reference = str(candidate)

        initial_state: LongVideoState = {
            "job_id": resolved_job_id,
            "request": request,
            "video_plan": None,
            "current_shot_index": 0,
            "completed_shot_paths": [],
            "completed_shots": [],
            "continuity_previous_last_frame": initial_reference,
            "continuity_reference_history": (
                [initial_reference]
                if initial_reference is not None
                else []
            ),
            "status": "planning",
            "final_output_path": None,
            "error": None,
        }

        result = await self.graph.ainvoke(initial_state)

        return result

    async def _plan_video(
        self,
        state: LongVideoState,
    ) -> dict:
        try:
            plan = await self._director.create_plan(
                state["request"]
            )

            return {
                "video_plan": plan,
                "current_shot_index": 0,
                "completed_shot_paths": [],
                "completed_shots": [],
                "status": "generating",
                "error": None,
            }

        except Exception as exc:
            return {
                "status": "failed",
                "error": (
                    f"Video planning failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    def _route_after_plan(
        self,
        state: LongVideoState,
    ) -> str:
        if state["status"] == "failed":
            return "failed"

        plan = state["video_plan"]

        if plan is None or not plan.shots:
            return "failed"

        return "generate"

    async def _generate_current_shot(
        self,
        state: LongVideoState,
    ) -> dict:
        plan = state["video_plan"]

        if plan is None:
            return {
                "status": "failed",
                "error": "Video plan is missing.",
            }

        index = state["current_shot_index"]

        if index >= len(plan.shots):
            return {
                "status": "composing",
            }

        shot = plan.shots[index]
        prompt = self._select_generation_prompt(shot)

        shot_dir = (
            self._output_dir
            / state["job_id"]
            / "shots"
        )

        shot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        requested_output_path = (
            shot_dir
            / f"{shot.shot_id}.mp4"
        )

        continuity = ShotContinuityState.from_json_state(
            previous_last_frame=state[
                "continuity_previous_last_frame"
            ],
            reference_history=state[
                "continuity_reference_history"
            ],
        )

        initial_reference = (
            continuity.previous_last_frame
            if self.continuity_enabled
            else None
        )

        try:
            generation_kwargs = {
                "shot": shot,
                "prompt": prompt,
                "output_path": requested_output_path,
            }

            # Preserve compatibility with older fake/test generators when
            # continuity is disabled.
            if self.continuity_enabled:
                generation_kwargs["initial_image"] = (
                    initial_reference
                )

            generated = await (
                self._shot_generator.generate_shot(
                    **generation_kwargs
                )
            )

            if isinstance(generated, GeneratedShot):
                resolved_path = generated.path.resolve()

                completed_info: CompletedShotInfo = {
                    "shot_id": shot.shot_id,
                    "path": str(resolved_path),
                    "requested_duration_seconds": (
                        generated.requested_duration_seconds
                    ),
                    "actual_duration_seconds": (
                        generated.actual_duration_seconds
                    ),
                    "duration_delta_seconds": (
                        generated.duration_delta_seconds
                    ),
                    "actual_fps": generated.fps,
                    "frame_count": generated.frame_count,
                    "width": generated.width,
                    "height": generated.height,
                    "initial_image_path": (
                        str(initial_reference)
                        if initial_reference is not None
                        else None
                    ),
                    "last_frame_reference_path": None,
                }
            else:
                resolved_path = Path(
                    generated
                ).resolve()

                completed_info = {
                    "shot_id": shot.shot_id,
                    "path": str(resolved_path),
                    "requested_duration_seconds": float(
                        shot.duration_seconds
                    ),
                    "actual_duration_seconds": None,
                    "duration_delta_seconds": None,
                    "actual_fps": None,
                    "frame_count": None,
                    "width": None,
                    "height": None,
                    "initial_image_path": (
                        str(initial_reference)
                        if initial_reference is not None
                        else None
                    ),
                    "last_frame_reference_path": None,
                }

            if not resolved_path.exists():
                raise FileNotFoundError(
                    "Shot generator returned a path "
                    "that does not exist: "
                    f"{resolved_path}"
                )

            if resolved_path.stat().st_size <= 0:
                raise RuntimeError(
                    "Shot generator produced an empty file: "
                    f"{resolved_path}"
                )

            continuity_update: dict[str, object] = {}

            if self._frame_extractor is not None:
                reference_dir = (
                    self._reference_asset_dir
                    / "jobs"
                    / state["job_id"]
                )

                reference_path = (
                    reference_dir
                    / f"{shot.shot_id}_last.png"
                )

                extraction = await (
                    self._frame_extractor.extract_last_frame(
                        resolved_path,
                        reference_path,
                    )
                )

                continuity = continuity.advance(
                    extraction.output_path
                )

                serialized = continuity.to_json_state()

                continuity_update = {
                    "continuity_previous_last_frame": (
                        serialized["previous_last_frame"]
                    ),
                    "continuity_reference_history": (
                        serialized["reference_history"]
                    ),
                }

                completed_info[
                    "last_frame_reference_path"
                ] = str(
                    extraction.output_path.resolve()
                )

            completed_paths = list(
                state["completed_shot_paths"]
            )
            completed_paths.append(
                str(resolved_path)
            )

            completed_shots = list(
                state["completed_shots"]
            )
            completed_shots.append(
                completed_info
            )

            return {
                "completed_shot_paths": completed_paths,
                "completed_shots": completed_shots,
                "current_shot_index": index + 1,
                "status": "generating",
                "error": None,
                **continuity_update,
            }

        except Exception as exc:
            return {
                "status": "failed",
                "error": (
                    f"Shot {shot.shot_id} generation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    def _route_after_generation(
        self,
        state: LongVideoState,
    ) -> str:
        if state["status"] == "failed":
            return "failed"

        plan = state["video_plan"]

        if plan is None:
            return "failed"

        if (
            state["current_shot_index"]
            < len(plan.shots)
        ):
            return "generate"

        return "compose"

    async def _compose_video(
        self,
        state: LongVideoState,
    ) -> dict:
        completed = state[
            "completed_shot_paths"
        ]

        if not completed:
            return {
                "status": "failed",
                "error": (
                    "No completed shots are available "
                    "for composition."
                ),
            }

        final_dir = (
            self._output_dir
            / state["job_id"]
        )

        final_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_output = (
            final_dir
            / "final.mp4"
        )

        try:
            result = await self._composer.concatenate(
                completed,
                final_output,
            )

            return {
                "status": "completed",
                "final_output_path": str(
                    Path(
                        result.output_path
                    ).resolve()
                ),
                "error": None,
            }

        except Exception as exc:
            return {
                "status": "failed",
                "error": (
                    f"Video composition failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    def _select_generation_prompt(
        self,
        shot: ShotPlan,
    ) -> str:
        if self._video_prompt_language == "zh":
            prompt = (
                shot.generation_prompt_zh
                or shot.prompt
                or shot.generation_prompt_en
            )
        else:
            prompt = (
                shot.generation_prompt_en
                or shot.prompt
                or shot.generation_prompt_zh
            )

        if not prompt or not prompt.strip():
            raise ValueError(
                f"Shot {shot.shot_id} has no usable "
                "generation prompt."
            )

        return prompt.strip()


__all__ = [
    "CompletedShotInfo",
    "LongVideoState",
    "LongVideoWorkflow",
    "ShotGenerator",
]
