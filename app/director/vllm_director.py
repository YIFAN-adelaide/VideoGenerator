from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import ValidationError

from app.director.base import BaseDirector, DirectorConfigurationError, DirectorPlanningError
from app.director.video_plan import DirectorRequest, VideoPlan

_SYSTEM_PROMPT = """
You are the Director and continuity supervisor for an AI video-generation system.

Convert the user's request into a structured long-video plan.

Rules:
1. Respect the requested total duration and maximum shot duration.
2. Split long videos into coherent short cinematic shots.
3. Preserve character identity, environment identity, lighting, visual style,
   spatial continuity, and narrative continuity across shots.
4. Keep machine-readable schema keys unchanged.
5. You may plan in English or Simplified Chinese according to the request.
6. EVERY shot must contain BOTH generation_prompt_en and generation_prompt_zh.
7. The English and Chinese prompts must describe the same shot.
8. Do not introduce unexplained character, clothing, location, time-of-day,
   or appearance changes between adjacent shots.
9. Camera changes must be intentional and cinematically coherent.
10. Return only data matching the supplied JSON schema.
""".strip()


class VLLMDirector(BaseDirector):
    """Director backed by a remote vLLM OpenAI-compatible server."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        temperature: float = 0.25,
        max_output_tokens: int = 4096,
        enable_thinking: bool = False,
    ) -> None:
        if not base_url.strip():
            raise DirectorConfigurationError("base_url cannot be empty.")
        if timeout_seconds <= 0:
            raise DirectorConfigurationError("timeout_seconds must be greater than 0.")
        if max_output_tokens <= 0:
            raise DirectorConfigurationError("max_output_tokens must be greater than 0.")

        super().__init__(director_name="vllm", model_name=model_name)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._enable_thinking = enable_thinking

    @property
    def chat_endpoint(self) -> str:
        if self._base_url.endswith("/v1"):
            return f"{self._base_url}/chat/completions"
        return f"{self._base_url}/v1/chat/completions"

    def build_payload(self, request: DirectorRequest) -> dict[str, Any]:
        user_payload = {
            "prompt": request.prompt,
            "target_duration_seconds": request.target_duration_seconds,
            "fps": request.fps,
            "resolution": request.resolution,
            "max_shot_duration_seconds": request.max_shot_duration_seconds,
            "user_language": request.user_language,
            "planning_language": request.planning_language,
        }

        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "video_plan",
                    "schema": VideoPlan.model_json_schema(),
                    "strict": True,
                },
            },
            "chat_template_kwargs": {
                "enable_thinking": self._enable_thinking,
            },
        }

    async def create_plan(self, request: DirectorRequest) -> VideoPlan:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(
                    self.chat_endpoint,
                    headers=headers,
                    json=self.build_payload(request),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise DirectorPlanningError(f"vLLM request failed: {exc}") from exc

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("vLLM returned empty message content.")
            return VideoPlan.model_validate_json(content)
        except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
            raise DirectorPlanningError(
                "vLLM returned an invalid VideoPlan response."
            ) from exc

    async def health(self) -> dict[str, Any]:
        base = self._base_url if self._base_url.endswith("/v1") else f"{self._base_url}/v1"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=min(self._timeout_seconds, 15.0)) as client:
                response = await client.get(f"{base}/models", headers=headers)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            return {
                "status": "error",
                "runtime": "vllm",
                "base_url": self._base_url,
                "model": self.model_name,
                "error": str(exc),
            }

        return {
            "status": "ok",
            "runtime": "vllm",
            "base_url": self._base_url,
            "model": self.model_name,
            "models_response": body,
        }
