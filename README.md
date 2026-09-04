# Image Conditioning v1

This patch adds the first user-image / continuity-image generation path.

## Design rule

The user's original image is never overwritten or stretched.

Instead:

```text
original image
    ↓
ImageProbe
    ↓
aspect-ratio-aware canvas resolver
    ↓
ImagePreprocessor (contain)
    ↓
model-ready PNG under reference_assets/
    ↓
VideoGenerationRequest
    ↓
FastVideoProvider
    ↓
FastVideo input_reference
```

The generated video canvas follows the uploaded image's width/height
proportion rather than forcing the existing landscape 480p/720p canvas.

Examples using the 720p compute profile:

```text
1024x1536  -> 768x1152   (2:3 portrait)
1920x1080  -> 1280x720   (16:9)
1280x704   -> 1280x704   (existing FastWan shape)
4000x1000  -> 1280x320   (4:1 panorama)
```

All dimensions are aligned to 16 for the current FastVideo/Wan path.

## Files

Add:

- `app/services/image_probe.py`
- `app/services/image_preprocessor.py`
- `tests/test_image_probe.py`
- `tests/test_image_preprocessor.py`
- `tests/test_fastvideo_image_conditioning.py`
- `scripts/smoke_image_to_video.py`

Update:

- `app/schemas.py`
- `app/providers/fastvideo.py`

## Local tests

```bash
pytest -q tests/test_image_probe.py
pytest -q tests/test_image_preprocessor.py
pytest -q tests/test_fastvideo_image_conditioning.py
pytest -q
```

## AWS shared input mount

FastVideo runs in Docker, while VideoGenerator runs on the host. The prepared
reference image therefore needs one shared directory.

Create it:

```bash
cd ~/VG/VideoGenerator
mkdir -p reference_assets/prepared
```

Restart the FastVideo container with this additional mount:

```bash
-v ~/VG/VideoGenerator/reference_assets:/inputs:ro
```

For your existing docker run command, that means adding:

```bash
-v ~/VG/VideoGenerator/reference_assets:/inputs:ro \
```

alongside the existing Hugging Face/output/config mounts.

The first generation after restarting the compiled FastVideo server may need
to compile/warm again.

## Real image-to-video test

You can use the final frame we already extracted as the source image:

```bash
python -m scripts.smoke_image_to_video \
  --image outputs/long-smoke-50df359b/references/shot_001_last.png \
  --prompt "The tiger continues walking naturally through the snowy forest." \
  --resolution 720p \
  --duration 5
```

The script will:

1. keep the original PNG untouched,
2. create a prepared image under `reference_assets/prepared/`,
3. preserve its aspect ratio,
4. pass `/inputs/prepared/...png` to FastVideo,
5. generate a 121-frame FastWan video,
6. probe and print the real output size.

For the current extracted 1280x704 frame, the canvas should remain 1280x704.

After this manual image-conditioned shot works, the next change is to connect
`FrameExtractor` automatically:

```text
Shot N
  ↓
FrameExtractor
  ↓
reference_assets/jobs/<job>/shot_N_last.png
  ↓
Shot N+1 initial_image
```
