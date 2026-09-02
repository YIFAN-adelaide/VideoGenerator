from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException

from app.config import settings
from app.graph.workflow import build_video_graph
from app.providers.factory import build_provider_resources
from app.schemas import (
    VideoGenerationRequest,
    VideoJobCreated,
    VideoJobStatus,
)
from app.services.job_store import InMemoryJobStore
from app.services.video_service import VideoService


resources = build_provider_resources(settings)
provider = resources.provider
graph = build_video_graph(provider)
store = InMemoryJobStore()

service = VideoService(
    graph=graph,
    store=store,
    max_concurrent_generations=settings.max_concurrent_generations,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Own long-lived model resources at the application boundary.

    Mock development:
        startup is effectively instant.

    AWS/Helios:
        load the model before FastAPI begins accepting traffic;
        unload it during graceful shutdown.
    """
    loader = resources.helios_loader

    if loader is not None:
        # Model loading is synchronous and can take a long time.
        # Keep it off the event loop during startup.
        await asyncio.to_thread(loader.load)

    try:
        yield
    finally:
        if loader is not None:
            await asyncio.to_thread(loader.unload)


app = FastAPI(
    title="Video Generator",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    response = {
        "status": "ok",
        "provider": settings.video_provider,
    }

    if resources.helios_loader is not None:
        response["runtime"] = resources.helios_loader.health()

    return response


@app.post(
    "/v1/videos",
    response_model=VideoJobCreated,
    status_code=202,
)
async def create_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
) -> VideoJobCreated:
    job_id = await service.submit(request)
    background_tasks.add_task(service.run, job_id)
    return VideoJobCreated(job_id=job_id)


@app.get(
    "/v1/videos/{job_id}",
    response_model=VideoJobStatus,
)
async def get_video(job_id: str) -> VideoJobStatus:
    state = await store.get(job_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail="Video job not found",
        )

    return VideoJobStatus(
        job_id=job_id,
        status=state["status"],
        progress=state["progress"],
        output_path=state.get("output_path"),
        error=state.get("error"),
        metadata=state.get("metadata", {}),
    )
