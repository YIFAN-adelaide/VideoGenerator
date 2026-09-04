import json

import httpx
import pytest

from app.providers.fastvideo import FastVideoProvider, FastVideoProviderError
from app.schemas import VideoGenerationRequest


@pytest.mark.asyncio
async def test_fastvideo_provider_submits_polls_and_downloads(tmp_path):
    seen_payload = {}
    status_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal status_calls

        if request.method == "POST" and request.url.path == "/v1/videos":
            seen_payload.update(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "id": "video-upstream-1",
                    "status": "queued",
                },
            )

        if (
            request.method == "GET"
            and request.url.path == "/v1/videos/video-upstream-1"
        ):
            status_calls += 1
            if status_calls == 1:
                return httpx.Response(
                    200,
                    json={
                        "id": "video-upstream-1",
                        "status": "in_progress",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "video-upstream-1",
                    "status": "completed",
                    "file_name": "result.mp4",
                    "timings": {"inference_seconds": 12.3},
                    "peak_memory_mb": 22000,
                },
            )

        if (
            request.method == "GET"
            and request.url.path
            == "/v1/videos/video-upstream-1/content"
        ):
            return httpx.Response(
                200,
                content=b"fake-fastvideo-mp4",
                headers={"content-type": "video/mp4"},
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url.path}"
        )

    client = httpx.AsyncClient(
        base_url="http://fastvideo.test",
        transport=httpx.MockTransport(handler),
    )
    provider = FastVideoProvider(
        base_url="http://fastvideo.test",
        model="FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        output_dir=str(tmp_path),
        poll_interval_seconds=0,
        client=client,
    )

    request = VideoGenerationRequest(
        prompt="A tiger walking through a snowy forest",
        duration_seconds=5,
        fps=24,
        resolution="720p",
        seed=123,
    )

    result = await provider.generate(request, "local-job-1")

    assert seen_payload == {
        "model": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        "prompt": "A tiger walking through a snowy forest",
        "seconds": 5.0,
        "fps": 24,
        "size": "1280x704",
        "seed": 123,
    }
    assert status_calls == 2
    assert result.output_path.endswith("local-job-1.mp4")
    assert (tmp_path / "local-job-1.mp4").read_bytes() == b"fake-fastvideo-mp4"
    assert result.metadata["provider"] == "fastvideo"
    assert result.metadata["upstream_job_id"] == "video-upstream-1"
    assert result.metadata["timings"] == {"inference_seconds": 12.3}
    assert result.metadata["peak_memory_mb"] == 22000

    await client.aclose()


@pytest.mark.asyncio
async def test_fastvideo_provider_surfaces_failed_job(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"id": "failed-1", "status": "queued"},
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "id": "failed-1",
                    "status": "failed",
                    "error": "out_of_memory",
                },
            )
        raise AssertionError("unexpected request")

    client = httpx.AsyncClient(
        base_url="http://fastvideo.test",
        transport=httpx.MockTransport(handler),
    )
    provider = FastVideoProvider(
        base_url="http://fastvideo.test",
        model="test-model",
        output_dir=str(tmp_path),
        poll_interval_seconds=0,
        client=client,
    )

    request = VideoGenerationRequest(
        prompt="test",
        duration_seconds=5,
        fps=24,
        resolution="720p",
    )

    with pytest.raises(FastVideoProviderError, match="out_of_memory"):
        await provider.generate(request, "local-job")

    await client.aclose()


@pytest.mark.asyncio
async def test_fastvideo_provider_health(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "workers": 1})

    client = httpx.AsyncClient(
        base_url="http://fastvideo.test",
        transport=httpx.MockTransport(handler),
    )
    provider = FastVideoProvider(
        base_url="http://fastvideo.test",
        model="test-model",
        output_dir=str(tmp_path),
        client=client,
    )

    health = await provider.health()

    assert health["status"] == "ok"
    assert health["runtime"] == "fastvideo_server"
    assert health["upstream"]["workers"] == 1

    await client.aclose()
