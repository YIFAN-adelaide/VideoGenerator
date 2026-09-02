import asyncio
from copy import deepcopy

from app.graph.state import VideoState


class InMemoryJobStore:
    """
    Phase-1 store only.

    Jobs disappear on process restart. We will replace this with durable
    persistence/checkpointing before treating the service as production-ready.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, VideoState] = {}
        self._lock = asyncio.Lock()

    async def create(self, state: VideoState) -> None:
        async with self._lock:
            self._jobs[state["job_id"]] = deepcopy(state)

    async def patch(self, job_id: str, changes: dict) -> None:
        async with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._jobs[job_id].update(deepcopy(changes))

    async def get(self, job_id: str) -> VideoState | None:
        async with self._lock:
            state = self._jobs.get(job_id)
            return deepcopy(state) if state is not None else None
