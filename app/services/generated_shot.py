from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GeneratedShot:
    """Observed result of generating one planned video shot."""

    path: Path
    requested_duration_seconds: float
    actual_duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.requested_duration_seconds <= 0:
            raise ValueError(
                "requested_duration_seconds must be greater than zero."
            )

        if self.actual_duration_seconds <= 0:
            raise ValueError(
                "actual_duration_seconds must be greater than zero."
            )

        if self.fps <= 0:
            raise ValueError("fps must be greater than zero.")

        if self.frame_count <= 0:
            raise ValueError(
                "frame_count must be greater than zero."
            )

        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "width and height must be greater than zero."
            )

    @property
    def duration_delta_seconds(self) -> float:
        """Actual duration minus the duration requested by the Director."""
        return (
            self.actual_duration_seconds
            - self.requested_duration_seconds
        )


__all__ = ["GeneratedShot"]
