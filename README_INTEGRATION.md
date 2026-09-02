# Helios lifecycle integration

This patch completes the application bootstrap layer.

## Startup flow

FastAPI process starts
→ provider resources are constructed
→ if VIDEO_PROVIDER=helios, lifespan calls loader.load()
→ only after load succeeds does FastAPI accept traffic

## Shutdown flow

SIGTERM / graceful shutdown
→ FastAPI lifespan exits
→ loader.unload()
→ CUDA/cache cleanup

## Local mode

VIDEO_PROVIDER=mock

No Helios/PyTorch/CUDA imports are required because the runtime loader remains
lazy and the lifespan has no GPU loader to start.

## AWS mode

VIDEO_PROVIDER=helios
HELIOS_REPO_PATH=/home/ubuntu/Helios
HELIOS_BASE_MODEL_PATH=BestWishYsh/Helios-Distilled
HELIOS_DEVICE=cuda:0
HELIOS_WEIGHT_DTYPE=bf16
HELIOS_LOW_VRAM=true
HELIOS_GROUP_OFFLOADING_TYPE=leaf_level

Run:

uvicorn app.main:app --host 127.0.0.1 --port 9100

The server will not become ready until Helios successfully loads.
