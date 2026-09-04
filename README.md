# FastVideo Wan I2V requested-canvas patch v1

This creates a reproducible custom FastVideo image for the current
VideoGenerator project.

## Why this patch exists

FastVideo 0.2.1 currently contains this Wan image-input validation rule:

```python
max_area = 480 * 832
ow, oh = best_output_size(iw, ih, dw, dh, max_area)
```

That means a VideoGenerator request for a 1280x704 image-conditioned shot can
still be reduced by FastVideo to a roughly 480p pixel budget.

This patch changes only the pixel budget:

```python
max_area = batch.height * batch.width
```

`best_output_size()` is intentionally retained. It is still responsible for
mapping the source aspect ratio onto valid Wan/VAE spatial alignment.

## Safety / reproducibility behavior

The patch script deliberately fails the Docker build if:

- FastVideo is not version 0.2.1
- the expected upstream source file is missing
- the exact hard-coded rule is no longer present exactly once
- the expected `best_output_size()` call is missing
- the patched Python source no longer compiles

This prevents silently carrying the patch onto a changed upstream release.

## Files

```text
docker/fastvideo/Dockerfile
docker/fastvideo/patch_wan_i2v_resolution.py
tests/test_fastvideo_i2v_resolution_patch.py
```

## Build on AWS

Pull the repository first, then:

```bash
cd ~/VG/VideoGenerator

docker build \
  -f docker/fastvideo/Dockerfile \
  -t videogen-fastvideo:0.2.1-i2v-resolution \
  .
```

The build log should include:

```text
FastVideo version: 0.2.1
Wan I2V resolution policy patched successfully.
Old: max_area = 480 * 832
New: max_area = batch.height * batch.width
```

## Replace the current FastVideo container

```bash
docker stop fastvideo-wan
docker rm fastvideo-wan
```

Start the custom image with the same current runtime settings:

```bash
docker run -d \
  --name fastvideo-wan \
  --gpus all \
  --ipc=host \
  -e FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN \
  -p 127.0.0.1:9200:9200 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/fastvideo_outputs:/outputs \
  -v ~/fastvideo_config:/configs:ro \
  -v ~/VG/VideoGenerator/reference_assets:/inputs:ro \
  videogen-fastvideo:0.2.1-i2v-resolution \
  /opt/venv/bin/fastvideo serve \
  --config /configs/fastwan5b_server.yaml
```

Wait for FastVideo:

```bash
docker logs -f fastvideo-wan
```

Use Ctrl+C to stop following logs, then verify health:

```bash
curl --fail-with-body http://127.0.0.1:9200/health
```

## Verify the running patch without GPU inference

```bash
pytest -q -s tests/test_fastvideo_i2v_resolution_patch.py
```

Expected context includes:

```python
max_area = batch.height * batch.width
```

and must not include:

```python
max_area = 480 * 832
```

## Re-run the existing real I2V smoke test

Use the existing extracted 1280x704 final frame:

```bash
python -m scripts.smoke_image_to_video \
  --image outputs/long-smoke-50df359b/references/shot_001_last.png \
  --prompt "The tiger continues walking naturally through the snowy forest." \
  --resolution 720p \
  --duration 5
```

For this test we want to see:

```text
source: 1280x704
canvas: 1280x704

video size: 1280x704
frames: 121
fps: 24.0
duration: 5.041667
```

The first generation after rebuilding/restarting the FastVideo image may pay
the compile/warm-up cost again.

## If the Docker build fails on the version guard

Do not remove the guard.

A failure means the upstream FastVideo image no longer matches the exact
0.2.1 implementation this patch was reviewed against. Inspect the new
`input_validation.py` before adapting the patch.
