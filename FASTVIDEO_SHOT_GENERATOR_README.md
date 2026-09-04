# FastVideoShotGenerator bridge

This patch adds the missing bridge between the tested LangGraph workflow and
the existing VideoProvider/FastVideoProvider contract.

```text
LongVideoWorkflow
       |
       v
FastVideoShotGenerator
       |
       v
VideoGenerationRequest
       |
       v
FastVideoProvider
       |
       v
FastVideo :9200
```

The adapter does not modify FastVideoProvider.

## Files

```text
app/services/fastvideo_shot_generator.py
app/tests/test_fastvideo_shot_generator.py
```

No existing `__init__.py` files are overwritten.

## Existing provider contract used by the adapter

```python
await provider.generate(
    request: VideoGenerationRequest,
    job_id: str,
)
```

`VideoGenerationRequest` fields used:

```text
prompt
duration_seconds
fps
resolution
seed
```

## Test locally

```powershell
pytest app/tests/test_fastvideo_shot_generator.py -v
pytest -q
```

## First real long-video acceptance target

After this adapter passes locally, deploy/pull the changes to the existing
L40S VideoGenerator instance. Keep the Director as `MockDirector` for the
first real run.

Target:

```text
15-second prompt
    |
    v
MockDirector
    |
    v
3 x 5-second ShotPlan
    |
    v
FastVideoShotGenerator
    |
    v
real FastVideoProvider
    |
    v
3 real MP4 shots
    |
    v
VideoComposer
    |
    v
final.mp4
```

Do not launch the second Director GPU until this path succeeds.
