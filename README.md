# Temporal Continuity V2.1

This patch tests a narrower hypothesis discovered from the V2 tiger video:

```text
Wrong behavior:
tiger reaches right edge
-> next clip places tiger back left
-> tiger repeats the traversal

Desired behavior:
tiger reaches right edge
-> camera pans/tracks with tiger
-> more forest is revealed
-> tiger keeps moving through world space
```

## Changes

### `TemporalContinuityState`
Adds:
- `subject_screen_behavior`
- `camera_response`
- `environment_reveal`

### `ContinuityPromptBuilder`
Explicitly tells FastWan:
- do not re-stage/reset the subject
- use camera response to follow the ongoing process
- reveal new environment when appropriate

### `LongVideoWorkflow`
For continuous Shot 2+, it prefers `ShotPlan.action` as the semantic continuation
prompt instead of the Director's standalone generation wrapper.

This is intended to remove conflicts such as:

```text
"Create cinematic shot 2 of 3"
"Camera: medium tracking shot"
```

versus:

```text
"preserve the current framing"
"continue directly from the previous frame"
```

Shot 1 still uses the Director's normal generation prompt.

## Local tests

```bash
pytest -q tests/test_continuity_prompt_builder_v2_1.py
pytest -q
```

## AWS

No Uvicorn restart is required for the smoke script.
FastVideo Docker must be running.

```bash
cd ~/VG/VideoGenerator
git pull
source .venv/bin/activate

set -a
source .env
set +a

curl --fail-with-body http://127.0.0.1:9200/health

python -m scripts.smoke_temporal_continuity_v2_1
```

Then compare V2.1 with V1 and V2, focusing on:
- whether the tiger resets to the left
- whether the camera follows the tiger
- whether new environment is revealed
- whether subject scale/framing stays stable
