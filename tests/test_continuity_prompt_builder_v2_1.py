from app.services.continuity_prompt_builder import (
    ContinuityPromptBuilder,
)
from app.services.temporal_continuity import (
    TemporalContinuityState,
)


def test_builder_can_tell_camera_to_follow_instead_of_reset_subject() -> None:
    state = TemporalContinuityState(
        ongoing_process=(
            "the tiger is already walking through the snowy forest"
        ),
        current_phase=(
            "the tiger has reached the right side of the current frame"
        ),
        next_development=(
            "continue the same walking action"
        ),
        spatial_change=(
            "the tiger continues forward through world space"
        ),
        subject_screen_behavior=(
            "do not reset the tiger to the left side; keep it naturally "
            "framed while it continues"
        ),
        camera_response=(
            "pan and track right with the tiger"
        ),
        environment_reveal=(
            "reveal additional snowy forest ahead on the right"
        ),
        preserve=(
            "the same tiger identity",
            "the same lighting and environment",
        ),
        avoid=(
            "restarting the tiger from the opposite side of the frame",
        ),
    )

    prompt = ContinuityPromptBuilder().build(
        base_prompt="The tiger continues walking.",
        state=state,
        has_previous_frame=True,
    )

    assert "pan and track right" in prompt
    assert "reveal additional snowy forest" in prompt
    assert "do not reset the tiger to the left side" in prompt
    assert "not the beginning of a new shot" in prompt


def test_builder_still_supports_non_locomotion_process() -> None:
    state = TemporalContinuityState(
        ongoing_process="the cardboard box is unfolding",
        current_phase="the side panels are partially open",
        next_development=(
            "continue rotating the panels outward until flatter"
        ),
        subject_screen_behavior=(
            "keep the partially unfolded box centered"
        ),
        camera_response="keep the camera static",
        preserve=("the same box and table",),
        avoid=("returning to the folded state",),
    )

    prompt = ContinuityPromptBuilder().build(
        base_prompt="The box continues unfolding.",
        state=state,
        has_previous_frame=True,
    )

    assert "partially unfolded box centered" in prompt
    assert "keep the camera static" in prompt
    assert state.spatial_change is None
