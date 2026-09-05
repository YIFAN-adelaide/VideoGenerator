from __future__ import annotations

from app.services.temporal_continuity import (
    TemporalContinuityState,
)


class ContinuityPromptBuilder:
    """
    Deterministic formatter.

    It does NOT call an LLM and it does NOT inspect the previous video.
    It converts structured temporal-continuity information produced by a
    Director/planner into explicit instructions for the video model.
    """

    def build(
        self,
        *,
        base_prompt: str,
        state: TemporalContinuityState,
        has_previous_frame: bool,
    ) -> str:
        cleaned = base_prompt.strip()

        if not cleaned:
            raise ValueError("base_prompt must not be empty")

        if state.mode == "hard_cut":
            return cleaned

        sections: list[str] = [cleaned]

        intro: list[str] = []

        if has_previous_frame:
            intro.append(
                "Continue directly from the supplied previous-frame "
                "reference. Treat this shot as the immediate next moment, "
                "not as the beginning of a new scene."
            )
        else:
            intro.append(
                "Treat this shot as a direct temporal continuation of the "
                "previous planned action."
            )

        sections.append(" ".join(intro))

        temporal_lines: list[str] = []

        if state.ongoing_process:
            temporal_lines.append(
                f"Ongoing process: {state.ongoing_process}."
            )

        if state.current_phase:
            temporal_lines.append(
                f"Current phase: {state.current_phase}."
            )

        if state.next_development:
            temporal_lines.append(
                f"Continue with: {state.next_development}."
            )

        if state.spatial_change:
            temporal_lines.append(
                f"Spatial continuity: {state.spatial_change}."
            )

        if state.orientation_change:
            temporal_lines.append(
                f"Orientation continuity: {state.orientation_change}."
            )

        if temporal_lines:
            sections.append(
                "Temporal continuity:\n- "
                + "\n- ".join(temporal_lines)
            )

        camera_lines: list[str] = []

        if state.camera_behavior:
            camera_lines.append(
                f"Camera behavior: {state.camera_behavior}."
            )

        if state.camera_framing:
            camera_lines.append(
                f"Framing: {state.camera_framing}."
            )

        if camera_lines:
            sections.append(
                "Camera continuity:\n- "
                + "\n- ".join(camera_lines)
            )

        if state.preserve:
            sections.append(
                "Preserve across the boundary:\n- "
                + "\n- ".join(
                    item.rstrip(".") + "."
                    for item in state.preserve
                )
            )

        if state.avoid:
            sections.append(
                "Avoid continuity breaks:\n- "
                + "\n- ".join(
                    item.rstrip(".") + "."
                    for item in state.avoid
                )
            )

        sections.append(
            "Maintain natural temporal progression. Do not reset the "
            "subject, object state, action, transformation, or camera unless "
            "the instructions above explicitly require a change."
        )

        return "\n\n".join(sections)


__all__ = ["ContinuityPromptBuilder"]
