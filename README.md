# Shot Continuity v1

This patch adds the first automatic long-video continuity loop:

```text
Shot 1
  ↓
generate MP4
  ↓
FrameExtractor
  ↓
reference_assets/jobs/<job>/shot_001_last.png
  ↓
Shot 2 initial_image
  ↓
generate MP4
  ↓
FrameExtractor
  ↓
shot_002_last.png
  ↓
Shot 3 initial_image
```

## Files

Add:

```text
app/services/shot_continuity.py
tests/test_fastvideo_shot_generator_continuity.py
tests/test_long_video_continuity.py
scripts/smoke_long_video_continuity.py
```

Update:

```text
app/services/fastvideo_shot_generator.py
app/graph/long_video_workflow.py
```

This patch assumes the previous milestones are already integrated:

- `FrameExtractor`
- `ImageProbe`
- `VideoGenerationRequest.initial_image`
- FastVideoProvider `input_reference`
- shared `reference_assets:/inputs:ro` Docker mount
- the FastVideo Wan I2V requested-canvas patch

## Backward compatibility

Continuity is enabled only when the workflow receives a `frame_extractor`.

Existing code:

```python
LongVideoWorkflow(
    director=...,
    shot_generator=...,
    composer=...,
    output_dir=...,
)
```

continues to run without continuity and does not pass `initial_image` to older
fake/test shot generators.

New continuity mode:

```python
LongVideoWorkflow(
    director=...,
    shot_generator=...,
    composer=...,
    output_dir=...,
    frame_extractor=FrameExtractor(),
    reference_asset_dir="reference_assets",
)
```

## Local tests

After copying the files into the repository:

```bash
pytest -q tests/test_fastvideo_shot_generator_continuity.py
pytest -q tests/test_long_video_continuity.py
pytest -q
```

AWS-only Docker diagnostics may show as skipped locally. That is expected.

## Commit/push

```bash
git add app/services/shot_continuity.py
git add app/services/fastvideo_shot_generator.py
git add app/graph/long_video_workflow.py
git add tests/test_fastvideo_shot_generator_continuity.py
git add tests/test_long_video_continuity.py
git add scripts/smoke_long_video_continuity.py

git commit -m "Add sequential shot continuity"
git push
```

## AWS

```bash
cd ~/VG/VideoGenerator
git pull
source .venv/bin/activate

set -a
source .env
set +a

pytest -q tests/test_fastvideo_shot_generator_continuity.py
pytest -q tests/test_long_video_continuity.py
```

Confirm the patched FastVideo server is still running:

```bash
curl --fail-with-body http://127.0.0.1:9200/health
```

Then run the real three-shot test:

```bash
python -m scripts.smoke_long_video_continuity
```

Expected logical handoff:

```text
shot_001 initial image: None
shot_001 last frame: reference_assets/jobs/<job>/shot_001_last.png

shot_002 initial image: .../shot_001_last.png
shot_002 last frame: .../shot_002_last.png

shot_003 initial image: .../shot_002_last.png
shot_003 last frame: .../shot_003_last.png
```

Each generated clip should remain 121 frames at 24fps (~5.041667 seconds).

## What this test proves

It proves the technical continuity chain. It does NOT yet prove that the model
produces perfect visual continuity.

After the real test, inspect:

1. the boundary between shot 1 and shot 2,
2. the boundary between shot 2 and shot 3,
3. whether the tiger identity/composition drifts,
4. whether the first frame of each new shot visually follows its reference.

Only after this is stable should we add:
- character reference bank
- environment/style references
- visual QC
- bounded retry
- transition/timeline editing
