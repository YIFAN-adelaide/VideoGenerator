# Long-Video LangGraph Workflow MVP

This patch adds the first orchestration core for long-form video generation.

## Placement

The current VideoGenerator project has a top-level `graph` package, so this
patch adds:

```text
graph/long_video_workflow.py
tests/test_long_video_workflow.py
```

It does not overwrite `graph/__init__.py`.

## Current graph

```text
START
  |
  v
plan_video
  |
  v
generate_current_shot
  |
  +---- more shots ----+
  |                    |
  +<-------------------+
  |
  v
compose_video
  |
  v
END
```

## Responsibility boundaries

- `BaseDirector` / `MockDirector`
  creates the `VideoPlan`
- `ShotGenerator`
  generates exactly one shot
- `VideoComposer`
  concatenates completed shots
- `LongVideoWorkflow`
  owns orchestration and state

## Why ShotGenerator is a Protocol

The current FastVideoProvider already works and should not be rewritten just
to satisfy the graph.

The graph therefore depends on this tiny adapter contract:

```python
async def generate_shot(
    *,
    shot: ShotPlan,
    prompt: str,
    output_path: Path,
) -> str | Path:
    ...
```

The next step is a small `FastVideoShotGenerator` adapter that translates this
contract into the existing `FastVideoProvider` request/result contract.

This keeps LangGraph independent of provider-specific details.

## Tests

```bash
pytest tests/test_long_video_workflow.py -v
pytest -q
```

The tests use:
- real `MockDirector`
- fake shot generator
- fake composer

No GPU, FastVideo server, or FFmpeg is required.

## First real AWS acceptance target

```text
15-second request
    ↓
MockDirector
    ↓
3 × 5-second ShotPlan
    ↓
FastVideoShotGenerator
    ↓
FastVideoProvider
    ↓
3 real MP4 files
    ↓
VideoComposer
    ↓
final.mp4
```

After that passes, reference-frame continuity and QC/retry can be added as
new LangGraph nodes without redesigning the graph.
