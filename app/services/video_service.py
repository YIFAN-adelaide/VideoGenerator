import asyncio
import uuid

from app.graph.state import VideoState
from app.schemas import VideoGenerationRequest
from app.services.job_store import InMemoryJobStore


class VideoService:
    def __init__(
        self,
        graph,
        store: InMemoryJobStore,
        max_concurrent_generations: int = 1,
    ) -> None:
        self.graph = graph
        self.store = store
        self._generation_slots = asyncio.Semaphore(max_concurrent_generations)

    async def submit(self, request: VideoGenerationRequest) -> str:
        job_id = uuid.uuid4().hex
        state: VideoState = {
            "job_id": job_id,
            "request": request.model_dump(),
            "status": "queued",
            "progress": 0.0,
            "output_path": None,
            "error": None,
            "metadata": {},
        }
        await self.store.create(state)
        return job_id

    async def run(self, job_id: str) -> None:
        state = await self.store.get(job_id)
        if state is None:
            return

        try:
            # One GPU should normally execute one generation at a time.
            async with self._generation_slots:
                async for graph_update in self.graph.astream(
                    state,
                    stream_mode="updates",
                ):
                    # updates has the shape: {"node_name": {"field": value, ...}}
                    for changes in graph_update.values():
                        if changes:
                            await self.store.patch(job_id, changes)
        except Exception as exc:
            await self.store.patch(
                job_id,
                {
                    "status": "failed",
                    "progress": 1.0,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
