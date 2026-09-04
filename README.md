# VideoProbe integration v1

This patch integrates the already-tested `VideoProbe` into the real long-video
generation path.

## Files

Add:
- `app/services/generated_shot.py`

Replace/update:
- `app/services/fastvideo_shot_generator.py`
- `app/graph/long_video_workflow.py`
- `app/tests/test_fastvideo_shot_generator.py`
- `app/tests/test_long_video_workflow.py`

Existing required file:
- `app/services/video_probe.py`

## What changes

`FastVideoShotGenerator.generate_shot()` now returns a `GeneratedShot`
containing the final MP4 path plus observed ffprobe metadata.

The LangGraph workflow remains backward-compatible with shot generators that
still return `Path` or `str`.

New state:
- `completed_shot_paths`: preserved for `VideoComposer`
- `completed_shots`: JSON-friendly per-shot observed metadata

For the current FastWan 5-second request, AWS should record approximately:

- requested duration: 5.0
- actual duration: 4.875
- delta: -0.125
- fps: 24
- frames: 117
- resolution: 1280x704

Three shots should therefore report about 14.625 seconds of observed generated
material.

## Test locally

```bash
pytest -q app/tests/test_video_probe.py
pytest -q app/tests/test_fastvideo_shot_generator.py
pytest -q app/tests/test_long_video_workflow.py
pytest -q
```

## AWS smoke test

After pushing and pulling:

```bash
cd ~/VG/VideoGenerator
source .venv/bin/activate
set -a
source .env
set +a

python -m scripts.smoke_long_video
```

To inspect the new metadata, temporarily add to the smoke script after
`result = await workflow.run(...)`:

```python
print("completed_shots:")
for shot in result["completed_shots"]:
    print(shot)

observed = sum(
    shot["actual_duration_seconds"] or 0.0
    for shot in result["completed_shots"]
)
print("observed generated duration:", observed)
```

Do not add an exact-duration trimming step yet. This patch is observability and
state integration only.
