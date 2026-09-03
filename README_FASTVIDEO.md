# FastVideo serving integration

The application now supports a separate FastVideo model-serving process.
FastVideo owns heavyweight model loading and GPU inference; VideoGenerator only
acts as the orchestration/API layer.

```text
client
  -> VideoGenerator FastAPI :9100
  -> LangGraph
  -> FastVideoProvider (HTTP)
  -> FastVideo server :9200
  -> GPU-resident video model
```

This avoids adding another custom PyTorch model loader to VideoGenerator.
`HeliosProvider` remains available as a reference implementation.

## Local development

Use `VIDEO_PROVIDER=mock` for normal laptop tests. The FastVideo provider is
unit-tested with an HTTP mock and therefore does not require CUDA or model
weights locally.

## AWS target

For the next benchmark use:

```text
VIDEO_PROVIDER=fastvideo
FASTVIDEO_BASE_URL=http://127.0.0.1:9200
FASTVIDEO_MODEL=FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers
FASTVIDEO_720P_SIZE=1280x704
```

FastVideo should be launched as its own long-lived server, bound to localhost.
Do not expose :9200 publicly. Our existing :9100 FastAPI service remains the
application boundary.

The first benchmark request should use `resolution="720p"`, because FastVideo's
current compatibility matrix validates FastWan2.2 TI2V 5B as a 720p-class
model. Model-specific denoising steps/guidance are intentionally not sent by
our provider so that FastVideo's served-model preset remains the source of
truth.
