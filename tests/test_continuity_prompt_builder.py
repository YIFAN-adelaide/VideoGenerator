from app.services.continuity_prompt_builder import (
    ContinuityPromptBuilder,
)
from app.services.temporal_continuity import (
    TemporalContinuityState,
)


def test_builder_supports_creature_motion() -> None:
    state = TemporalContinuityState(
        ongoing_process=(
            "the tiger is walking steadily through the forest"
        ),
        next_development=(
            "continue the same walking action without stopping"
        ),
        spatial_change=(
            "continue forward-right at approximately the same pace"
        ),
        camera_behavior=(
            "continue the same tracking movement"
        ),
        camera_framing="maintain a medium shot",
        preserve=(
            "the same tiger identity and fur pattern",
            "the same snowy forest and lighting",
        ),
        avoid=(
            "repositioning the tiger to a new starting point",
            "restarting the walking cycle as a new scene",
        ),
    )

    prompt = ContinuityPromptBuilder().build(
        base_prompt=(
            "The tiger continues deeper into the snowy forest."
        ),
        state=state,
        has_previous_frame=True,
    )

    assert "forward-right" in prompt
    assert "previous-frame reference" in prompt
    assert "repositioning" in prompt
    assert "tracking movement" in prompt


def test_builder_supports_non_locomotion_transformation() -> None:
    state = TemporalContinuityState(
        ongoing_process=(
            "the cardboard box is unfolding"
        ),
        current_phase=(
            "the side panels are already partially open"
        ),
        next_development=(
            "continue rotating the remaining panels outward "
            "until the box becomes flatter"
        ),
        camera_behavior="keep the camera static",
        preserve=(
            "the same box",
            "the same table and lighting",
        ),
        avoid=(
            "returning the box to the folded state",
            "resetting the box position",
        ),
    )

    prompt = ContinuityPromptBuilder().build(
        base_prompt=(
            "The cardboard box continues unfolding."
        ),
        state=state,
        has_previous_frame=True,
    )

    assert "unfolding" in prompt
    assert "partially open" in prompt
    assert "becomes flatter" in prompt
    assert "folded state" in prompt

    # No locomotion-specific field is required.
    assert state.spatial_change is None


def test_hard_cut_does_not_force_continuity() -> None:
    base = "A wide aerial shot of the mountain."

    state = TemporalContinuityState(
        mode="hard_cut",
        ongoing_process="irrelevant for hard cut",
    )

    prompt = ContinuityPromptBuilder().build(
        base_prompt=base,
        state=state,
        has_previous_frame=True,
    )

    assert prompt == base
