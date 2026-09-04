# Bilingual Director V2

This patch upgrades the Director contract so planning can work in English or Chinese while every shot carries both an English and Simplified Chinese video-model prompt.

## Main changes

- bilingual `DirectorRequest`
- structured character/environment/style profiles
- bilingual `ShotPlan` prompts
- V1-compatible `shot.prompt`
- bilingual `MockDirector`
- `VLLMDirector` using vLLM's OpenAI-compatible `/v1/chat/completions`
- JSON Schema structured output
- Qwen3 thinking disabled by default for predictable planning

## Dependency

The application-side vLLM client uses `httpx`. Do not install vLLM itself into the VideoGenerator environment.

```bash
python -m pip install httpx
```

## Tests

```bash
pytest tests/test_video_plan.py tests/test_mock_director.py tests/test_vllm_director.py -v
pytest -q
```

## Future runtime

```text
VideoGenerator / LangGraph
        |
        +--> VLLMDirector --> private VPC --> vLLM :9300 --> Director LLM
        |
        +--> FastVideoProvider --> FastVideo :9200 --> FastWan
```

Keep the Director's port private. Qwen3 thinking can later be enabled selectively for difficult planning or QC tasks.
