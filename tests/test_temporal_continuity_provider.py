import asyncio
from types import SimpleNamespace

from app.services.temporal_continuity import (
    TemporalContinuityState,
)
from app.services.temporal_continuity_provider import (
    StaticTemporalContinuityProvider,
)


def test_static_provider_uses_destination_shot_id() -> None:
    state = TemporalContinuityState(
        ongoing_process="continue walking",
    )

    provider = StaticTemporalContinuityProvider(
        {
            "shot_002": state,
        }
    )

    result = asyncio.run(
        provider.describe_transition(
            request=SimpleNamespace(),
            previous_shot=SimpleNamespace(
                shot_id="shot_001"
            ),
            current_shot=SimpleNamespace(
                shot_id="shot_002"
            ),
            current_shot_index=1,
        )
    )

    assert result is state
