import asyncio

from app.director.mock_director import MockDirector
from app.director.video_plan import DirectorRequest


def test_mock_director_splits_15_seconds_into_three_shots():
    plan = asyncio.run(
        MockDirector().create_plan(
            DirectorRequest(
                prompt="A tiger explores a snowy forest.",
                target_duration_seconds=15,
                max_shot_duration_seconds=5,
            )
        )
    )
    assert [s.duration_seconds for s in plan.shots] == [5, 5, 5]
    assert plan.planned_duration_seconds == 15


def test_mock_director_detects_chinese():
    plan = asyncio.run(
        MockDirector().create_plan(
            DirectorRequest(
                prompt="一只小老虎从山洞走进雪地森林。",
                target_duration_seconds=10,
            )
        )
    )
    assert plan.detected_language == "zh"
    assert plan.planning_language == "zh"


def test_mock_director_detects_english():
    plan = asyncio.run(
        MockDirector().create_plan(
            DirectorRequest(
                prompt="A tiger walks through a snowy forest.",
                target_duration_seconds=10,
            )
        )
    )
    assert plan.detected_language == "en"
    assert plan.planning_language == "en"


def test_mock_director_populates_both_prompt_languages():
    plan = asyncio.run(
        MockDirector().create_plan(
            DirectorRequest(
                prompt="一只小老虎在雪地里散步。",
                target_duration_seconds=5,
            )
        )
    )
    shot = plan.shots[0]
    assert "Create cinematic shot" in shot.generation_prompt_en
    assert "电影感镜头" in shot.generation_prompt_zh
