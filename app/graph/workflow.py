from langgraph.graph import END, START, StateGraph

from app.graph.state import VideoState
from app.providers.base import VideoProvider
from app.schemas import VideoGenerationRequest


def build_video_graph(provider: VideoProvider):
    async def prepare(state: VideoState) -> dict:
        # Pydantic validation here also protects non-HTTP graph callers.
        request = VideoGenerationRequest.model_validate(state["request"])
        return {
            "request": request.model_dump(),
            "status": "preparing",
            "progress": 0.10,
            "error": None,
        }

    async def generate(state: VideoState) -> dict:
        request = VideoGenerationRequest.model_validate(state["request"])
        result = await provider.generate(request, state["job_id"])
        return {
            "status": "generating",
            "progress": 0.90,
            "output_path": result.output_path,
            "metadata": result.metadata,
        }

    async def finalize(state: VideoState) -> dict:
        return {
            "status": "completed",
            "progress": 1.0,
        }

    builder = StateGraph(VideoState)
    builder.add_node("prepare", prepare)
    builder.add_node("generate", generate)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "generate")
    builder.add_edge("generate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
