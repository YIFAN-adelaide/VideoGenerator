import pytest

from app.graph.workflow import build_video_graph
from app.providers.mock import MockVideoProvider


@pytest.mark.asyncio
async def test_graph_completes(tmp_path):
    provider = MockVideoProvider(str(tmp_path))
    graph = build_video_graph(provider)

    result = await graph.ainvoke(
        {
            "job_id": "test-job",
            "request": {
                "prompt": "A tiger walking through a forest",
                "duration_seconds": 4,
                "fps": 24,
                "resolution": "480p",
                "seed": 123,
            },
            "status": "queued",
            "progress": 0.0,
            "output_path": None,
            "error": None,
            "metadata": {},
        }
    )

    assert result["status"] == "completed"
    assert result["progress"] == 1.0
    assert result["output_path"]
