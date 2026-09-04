import pytest
from pydantic import ValidationError

from app.director.video_plan import DirectorRequest, ShotPlan, VideoPlan


def test_director_request_accepts_language_settings():
    request = DirectorRequest(
        prompt="一只小老虎走过雪地森林。",
        target_duration_seconds=15,
        user_language="zh",
        planning_language="zh",
    )
    assert request.user_language == "zh"
    assert request.planning_language == "zh"
    assert request.max_shot_duration_seconds == 5


def test_director_request_rejects_empty_prompt():
    with pytest.raises(ValidationError):
        DirectorRequest(prompt="   ", target_duration_seconds=15)


def test_shot_plan_keeps_v1_prompt_compatibility():
    shot = ShotPlan(
        shot_id="shot_001",
        prompt="A cinematic tiger in snow.",
        duration_seconds=5,
    )
    assert shot.generation_prompt_en == shot.prompt
    assert shot.generation_prompt_zh == shot.prompt


def test_shot_plan_accepts_bilingual_prompts():
    shot = ShotPlan(
        shot_id="shot_001",
        generation_prompt_en="A tiger walks through snow.",
        generation_prompt_zh="一只老虎走过雪地。",
        duration_seconds=5,
    )
    assert shot.generation_prompt_en == "A tiger walks through snow."
    assert shot.generation_prompt_zh == "一只老虎走过雪地。"
    assert shot.prompt == "A tiger walks through snow."


def test_video_plan_calculates_planned_duration():
    plan = VideoPlan(
        original_prompt="Test",
        target_duration_seconds=12,
        shots=[
            ShotPlan(shot_id="shot_001", prompt="Scene one", duration_seconds=5),
            ShotPlan(shot_id="shot_002", prompt="Scene two", duration_seconds=5),
            ShotPlan(shot_id="shot_003", prompt="Scene three", duration_seconds=2),
        ],
    )
    assert plan.planned_duration_seconds == 12


def test_video_plan_schema_can_be_generated_for_vllm():
    schema = VideoPlan.model_json_schema()
    assert schema["type"] == "object"
    assert "shots" in schema["properties"]
