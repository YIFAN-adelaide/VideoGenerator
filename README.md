# Frame Extractor v1

Adds exact frame extraction for continuity/reference workflows.

## Files

- `app/services/frame_extractor.py`
- `tests/test_frame_extractor.py`

## Design

`FrameExtractor.extract_last_frame()`:

1. probes the source MP4 with `VideoProbe`
2. reads the real decoded frame count
3. calculates `last_index = frame_count - 1`
4. asks ffmpeg to select that exact frame
5. writes a lossless PNG reference image

For the current FastWan output:

```text
frame_count = 121
last_index  = 120
```

The ffmpeg filter is therefore:

```text
select=eq(n\,120)
```

This deliberately avoids approximate `-sseof` timestamp seeking.

## Local test

```bash
pytest -q tests/test_frame_extractor.py
pytest -q
```

## AWS real-file test

After pushing/pulling, use an existing generated shot:

```bash
python - <<'PY'
import asyncio
from pathlib import Path

from app.services.frame_extractor import FrameExtractor

VIDEO = Path(
    "outputs/long-smoke-50df359b/"
    "shots/shot_001.mp4"
)

OUTPUT = Path(
    "outputs/long-smoke-50df359b/"
    "references/shot_001_last.png"
)

async def main():
    result = await FrameExtractor().extract_last_frame(
        VIDEO,
        OUTPUT,
    )

    print("source:", result.source_path)
    print("output:", result.output_path)
    print("frame_index:", result.frame_index)
    print("source_frame_count:", result.source_frame_count)
    print("fps:", result.source_fps)
    print("duration:", result.source_duration_seconds)

asyncio.run(main())
PY
```

Expected:

```text
frame_index: 120
source_frame_count: 121
fps: 24.0
duration: 5.041667
```

Then verify the PNG exists:

```bash
file outputs/long-smoke-50df359b/references/shot_001_last.png
ls -lh outputs/long-smoke-50df359b/references/shot_001_last.png
```

This stage only extracts reference assets. It does not yet pass them back to
FastWan. The next milestone is adding an optional `initial_image` to the
shot-generation request/provider contract.
