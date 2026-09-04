from app.director.vllm_director import VLLMDirector
from app.director.video_plan import DirectorRequest


def test_vllm_director_builds_endpoint():
    director = VLLMDirector(
        base_url="http://10.0.0.20:9300",
        model_name="Qwen/Qwen3-8B",
    )
    assert director.chat_endpoint == "http://10.0.0.20:9300/v1/chat/completions"


def test_vllm_director_accepts_base_url_with_v1():
    director = VLLMDirector(
        base_url="http://10.0.0.20:9300/v1",
        model_name="Qwen/Qwen3-8B",
    )
    assert director.chat_endpoint == "http://10.0.0.20:9300/v1/chat/completions"


def test_vllm_director_requests_json_schema_output():
    director = VLLMDirector(
        base_url="http://10.0.0.20:9300",
        model_name="Qwen/Qwen3-8B",
        enable_thinking=False,
    )
    payload = director.build_payload(
        DirectorRequest(
            prompt="一只小老虎走过雪地森林。",
            target_duration_seconds=15,
        )
    )
    assert payload["response_format"]["type"] == "json_schema"
    assert "shots" in payload["response_format"]["json_schema"]["schema"]["properties"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "一只小老虎走过雪地森林。" in payload["messages"][1]["content"]
