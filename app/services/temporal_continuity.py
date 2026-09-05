from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ContinuityMode = Literal[
    "continuous",
    "soft_transition",
    "hard_cut",
]


@dataclass(frozen=True, slots=True)
class TemporalContinuityState:
    """
    Semantic description of what must continue across one shot boundary.

    This is intentionally more general than "motion continuity". It supports:
    - locomotion: a tiger keeps walking
    - transformations: a box keeps unfolding
    - processes: ice keeps melting
    - object interaction: a hand keeps writing
    - camera continuity

    Every field is optional. The Director should populate only what is
    meaningful for the current transition.
    """

    mode: ContinuityMode = "continuous"

    ongoing_process: str | None = None
    current_phase: str | None = None
    next_development: str | None = None

    spatial_change: str | None = None
    orientation_change: str | None = None

    camera_behavior: str | None = None
    camera_framing: str | None = None

    preserve: tuple[str, ...] = field(default_factory=tuple)
    avoid: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "ongoing_process": self.ongoing_process,
            "current_phase": self.current_phase,
            "next_development": self.next_development,
            "spatial_change": self.spatial_change,
            "orientation_change": self.orientation_change,
            "camera_behavior": self.camera_behavior,
            "camera_framing": self.camera_framing,
            "preserve": list(self.preserve),
            "avoid": list(self.avoid),
        }


__all__ = [
    "ContinuityMode",
    "TemporalContinuityState",
]
