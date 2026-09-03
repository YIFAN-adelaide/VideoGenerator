# FastVideo provider patch

## What changed

- Added `app/providers/fastvideo.py`.
  - Uses FastVideo's async `/v1/videos` API.
  - Polls `/v1/videos/{id}` until completion.
  - Streams `/v1/videos/{id}/content` to the application's output directory.
  - Exposes FastVideo `/health` through the application health endpoint.
  - Keeps FastVideo/model loading completely outside the VideoGenerator process.
- Added `VIDEO_PROVIDER=fastvideo` to the provider factory.
- Added FastVideo settings to `app/config.py` and `.env.example`.
- Added optional `health()` and `close()` lifecycle hooks to `VideoProvider`.
- Fixed LangGraph status ordering by publishing `generating` before the expensive
  provider call starts.
- Added FastVideo provider/factory tests.

## Local verification

The FastVideo adapter, factory, and existing Helios provider/runtime tests pass
without CUDA:

```text
18 passed
```

`python -m compileall -q app tests` also passes.

The complete suite could not be run in this artifact environment because its
Python environment does not have `langgraph` installed and has no package
network access. The project requirements already include LangGraph, so run the
full test suite inside the project's normal Python 3.11 environment.

## First AWS target

Use a long-lived FastVideo server on localhost port 9200 and our application on
localhost port 9100. Start with:

```text
FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers
```

For the first benchmark, send `resolution="720p"`. The application maps this
to `1280x704` by default, which matches Wan2.2 TI2V-5B's native landscape
shape. FastVideo remains responsible for the distilled model's sampling preset;
the adapter intentionally does not override inference steps or guidance.
