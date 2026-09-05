# Temporal Continuity V2

This patch tests the next hypothesis:

```text
V1
previous final frame
    -> FastWan

V2
previous final frame
+ semantic TemporalContinuityState
+ deterministic ContinuityPromptBuilder
    -> FastWan
```

## Important architecture decision

`ContinuityPromptBuilder` does not call another model.

For this experiment, a `StaticTemporalContinuityProvider` supplies known-good
transition instructions. Later the vLLM/Qwen Director can implement the same
`TemporalContinuityProvider` protocol and generate these states automatically.

That means the user will NOT manually enter these fields in the final system.

## Add

```text
app/services/temporal_continuity.py
app/services/continuity_prompt_builder.py
app/services/temporal_continuity_provider.py
tests/test_continuity_prompt_builder.py
tests/test_temporal_continuity_provider.py
scripts/smoke_temporal_continuity_v2.py
```

## Update

```text
app/graph/long_video_workflow.py
```

No change to FastVideoShotGenerator is required if Shot Continuity V1 is
already integrated.

## Local

```bash
pytest -q tests/test_continuity_prompt_builder.py
pytest -q tests/test_temporal_continuity_provider.py
pytest -q
```

## Commit

```bash
git add app/services/temporal_continuity.py
git add app/services/continuity_prompt_builder.py
git add app/services/temporal_continuity_provider.py
git add app/graph/long_video_workflow.py
git add tests/test_continuity_prompt_builder.py
git add tests/test_temporal_continuity_provider.py
git add scripts/smoke_temporal_continuity_v2.py

git commit -m "Add semantic temporal continuity prompts"
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

pytest -q tests/test_continuity_prompt_builder.py
pytest -q tests/test_temporal_continuity_provider.py

curl --fail-with-body http://127.0.0.1:9200/health
```

Run:

```bash
python -m scripts.smoke_temporal_continuity_v2
```

The script prints the exact effective FastWan prompt for each shot.

Download the resulting `final.mp4` and compare it with the previous V1
continuity video. Focus on:

- subject screen-position jump
- direction reset
- action/gait reset
- camera reframing
- identity/environment continuity

## Why the test provider is static

We intentionally do not integrate vLLM in the same experiment. If V2 improves
motion continuity, we have isolated the benefit of the semantic state and
prompt builder. Then the next patch is to make the real Director generate the
same structured state.
