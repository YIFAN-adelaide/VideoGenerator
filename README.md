# FastWan duration fix v1

This patch fixes the current *undershoot* problem for FastWan2.2 TI2V 5B.

Observed before:
- request: 5.0 seconds @ 24 FPS
- output: 117 frames
- actual: 4.875 seconds

New provider behavior:
- semantic target: 5.0 seconds
- requested frames: 120
- provider aligns upward to the next `4n + 1` frame count
- FastVideo payload sends `num_frames=121` explicitly
- expected raw generated duration: about 5.0417 seconds

Why upward?
A later editing stage can drop/trim excess frames. It cannot recover a
shortfall without duplicating/interpolating frames.

## Files

Add:
- `app/providers/fastvideo_duration.py`

Replace/update:
- `app/providers/fastvideo.py`

Add tests:
- `tests/test_fastvideo_duration.py`
- `tests/test_fastvideo_duration_payload.py`

## Run locally

```bash
pytest -q tests/test_fastvideo_duration.py
pytest -q tests/test_fastvideo_duration_payload.py
pytest -q
```

## Important

This is phase 1: prevent undershoot.

Do NOT add trimming yet. First push this to AWS and run one real smoke test.
The expected real shot metadata should become approximately:

```text
requested_duration_seconds: 5.0
actual_duration_seconds:     5.041667
frame_count:                 121
fps:                         24
```

Once that is confirmed, phase 2 is an exact-duration trim node/service that
removes the one excess frame so each 5-second production shot becomes exactly
120 frames at 24 FPS.
