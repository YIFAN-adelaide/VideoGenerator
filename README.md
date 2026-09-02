# VideoGenerator

Phase 1 of a local-development / AWS-GPU video generation service.

## Current architecture

FastAPI -> LangGraph -> VideoProvider

Local development uses `MockVideoProvider`.
AWS will later use `HeliosProvider`.

The first graph is deliberately small:

START -> prepare -> generate -> finalize -> END

This proves the API and orchestration contract before Helios/CUDA/model
dependencies are introduced.

## Python

Python 3.11 is recommended.

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

## Start locally

PowerShell:

```powershell
$env:VIDEO_PROVIDER="mock"
uvicorn app.main:app --host 127.0.0.1 --port 9100 --reload
```

Open:

- http://127.0.0.1:9100/docs
- http://127.0.0.1:9100/health

## Create a mock generation job

```powershell
$body = @{
  prompt = "A cinematic tiger walking through a forest"
  duration_seconds = 4
  fps = 24
  resolution = "480p"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9100/v1/videos" `
  -ContentType "application/json" `
  -Body $body
```

Then poll:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:9100/v1/videos/<JOB_ID>"
```

## AWS port

The service will still listen on `127.0.0.1:9100` on EC2 and be reached
through the SSH tunnel:

```powershell
ssh -i "D:\VideoGenerator\Documents\Video_Generator.pem" `
  -L 9100:127.0.0.1:9100 `
  ubuntu@<EC2_PUBLIC_IP>
```

Do not expose port 9100 publicly.

## Next implementation phase

1. Implement `HeliosProvider`.
2. Normalize requested duration to Helios frame/chunk constraints.
3. Add real progress/metrics around model inference.
4. Add durable LangGraph checkpointing/job persistence.
5. Add LangChain LLM scene planning for long videos.
6. Add multi-scene generation + FFmpeg composition.
