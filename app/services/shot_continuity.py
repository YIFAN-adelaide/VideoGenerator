from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ShotContinuityState:
    """
    Minimal continuity state for sequential long-video generation.

    V1 deliberately tracks only the previous shot's final frame. Character,
    environment, style, and master-reference banks can be added later without
    changing the basic shot-to-shot handoff.
    """

    previous_last_frame: Path | None = None
    reference_history: tuple[Path, ...] = ()

    def advance(
        self,
        last_frame: str | Path,
    ) -> "ShotContinuityState":
        resolved = Path(last_frame).expanduser().resolve()

        if not resolved.exists():
            raise FileNotFoundError(
                f"Continuity reference does not exist: {resolved}"
            )

        if not resolved.is_file():
            raise ValueError(
                f"Continuity reference is not a file: {resolved}"
            )

        if resolved.stat().st_size <= 0:
            raise ValueError(
                f"Continuity reference is empty: {resolved}"
            )

        return ShotContinuityState(
            previous_last_frame=resolved,
            reference_history=(
                *self.reference_history,
                resolved,
            ),
        )

    def to_json_state(self) -> dict[str, object]:
        return {
            "previous_last_frame": (
                str(self.previous_last_frame)
                if self.previous_last_frame is not None
                else None
            ),
            "reference_history": [
                str(path)
                for path in self.reference_history
            ],
        }

    @classmethod
    def from_json_state(
        cls,
        *,
        previous_last_frame: str | None,
        reference_history: list[str],
    ) -> "ShotContinuityState":
        return cls(
            previous_last_frame=(
                Path(previous_last_frame).resolve()
                if previous_last_frame
                else None
            ),
            reference_history=tuple(
                Path(path).resolve()
                for path in reference_history
            ),
        )


__all__ = ["ShotContinuityState"]
