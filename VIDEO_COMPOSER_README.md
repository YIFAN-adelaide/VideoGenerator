# VideoComposer MVP

This component joins compatible FastWan/FastVideo MP4 shots into one final MP4.

It uses FFmpeg concat with `-c copy`, so V1 has:
- no re-encoding
- no quality loss from composition
- low CPU overhead
- fast concatenation

Input clips must use compatible stream settings. That matches our current
long-video MVP because every shot will use the same model, resolution, FPS,
and provider profile.

## Add files

Copy:
- `app/services/video_composer.py`
- `tests/test_video_composer.py`

This patch intentionally does not overwrite an existing
`app/services/__init__.py`.

## Test

```bash
pytest tests/test_video_composer.py -v
pytest -q
```

The unit tests mock the FFmpeg process, so they do not require FFmpeg.

## Runtime requirement

On Windows:

```powershell
ffmpeg -version
```

On Ubuntu/AWS:

```bash
ffmpeg -version
```

If missing on Ubuntu:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Example

```python
from app.services.video_composer import VideoComposer

composer = VideoComposer()

result = await composer.concatenate(
    [
        "outputs/shot_001.mp4",
        "outputs/shot_002.mp4",
        "outputs/shot_003.mp4",
    ],
    "outputs/final.mp4",
)
```
