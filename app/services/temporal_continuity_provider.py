from __future__ import annotations

from typing import Protocol

from app.director.video_plan import (
    DirectorRequest,
    ShotPlan,
)
from app.services.temporal_continuity import (
    TemporalContinuityState,
)


class TemporalContinuityProvider(Protocol):
    """
    Produces semantic transition state for the boundary leading into
    ``current_shot``.

    V2 tests can use StaticTemporalContinuityProvider.
    A future vLLM Director implementation can implement this same protocol.
    """

    async def describe_transition(
        self,
        *,
        request: DirectorRequest,
        previous_shot: ShotPlan,
        current_shot: ShotPlan,
        current_shot_index: int,
    ) -> TemporalContinuityState | None:
        ...


class StaticTemporalContinuityProvider:
    """
    Deterministic provider for experiments/tests.

    ``by_current_shot_id`` maps the destination shot ID to the state that
    should be used when entering that shot.
    """

    def __init__(
        self,
        by_current_shot_id: dict[
            str,
            TemporalContinuityState,
        ],
    ) -> None:
        self._states = dict(by_current_shot_id)

    async def describe_transition(
        self,
        *,
        request: DirectorRequest,
        previous_shot: ShotPlan,
        current_shot: ShotPlan,
        current_shot_index: int,
    ) -> TemporalContinuityState | None:
        del request
        del previous_shot
        del current_shot_index

        return self._states.get(
            current_shot.shot_id
        )


__all__ = [
    "StaticTemporalContinuityProvider",
    "TemporalContinuityProvider",
]
