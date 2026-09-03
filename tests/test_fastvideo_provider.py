import asyncio

import pytest

from app.graph.workflow import build_video_graph
from app.providers.mock import MockVideoProvider


@pytest.mark.asyncio
async def test_graph_marks_generating_before_provider_call(tmp_path):
    provider_entered = asyncio.Event()
    provider_release = asyncio.Event()

    class BlockingProvider(MockVideoProvider):
        async def generate(self, request, job_id):
            provider_entered.set()

            await provider_release.wait()

            return await super().generate(
                request,
                job_id,
            )

    provider = BlockingProvider(str(tmp_path))
    graph = build_video_graph(provider)

    state = {
        "job_id": "test-job",
        "request": {
            "prompt": "test",
            "duration_seconds": 1,
            "fps": 24,
            "resolution": "480p",
            "seed": 1,
        },
        "status": "queued",
        "progress": 0.0,
        "output_path": None,
        "error": None,
        "metadata": {},
    }

    stream = graph.astream(
        state,
        stream_mode="updates",
    )

    first = await anext(stream)
    second = await anext(stream)

    assert first["prepare"]["status"] == "preparing"
    assert second["mark_generating"]["status"] == "generating"

    # Start execution of the next graph node.
    next_update = asyncio.create_task(
        anext(stream)
    )

    # Wait deterministically until the provider has actually been entered.
    await asyncio.wait_for(
        provider_entered.wait(),
        timeout=1.0,
    )

    # The provider is blocked, so the generate node cannot have returned yet.
    assert next_update.done() is False

    provider_release.set()

    generate_update = await next_update

    assert "generate" in generate_update
    assert generate_update["generate"]["status"] == "generating"

    await stream.aclose()