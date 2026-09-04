from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

LanguageCode = Literal["auto", "en", "zh"]


class DirectorRequest(BaseModel):
    """Provider-independent request sent to a video Director."""

    prompt: str = Field(min_length=1)
    target_duration_seconds: float = Field(gt=0)
    fps: int = Field(default=24, gt=0)
    resolution: str = Field(default="720p", min_length=1)
    max_shot_duration_seconds: float = Field(default=5.0, gt=0)
    user_language: LanguageCode = "auto"
    planning_language: LanguageCode = "auto"

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt cannot be empty.")
        return value


class CharacterProfile(BaseModel):
    character_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    appearance: str = Field(min_length=1)
    distinctive_features: list[str] = Field(default_factory=list)
    clothing: str | None = None
    consistency_rules: list[str] = Field(default_factory=list)


class EnvironmentProfile(BaseModel):
    environment_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    visual_features: list[str] = Field(default_factory=list)
    time_of_day: str | None = None
    weather: str | None = None
    consistency_rules: list[str] = Field(default_factory=list)


class VisualStyle(BaseModel):
    look: str | None = None
    lens: str | None = None
    color_palette: str | None = None
    camera_motion: str | None = None
    lighting_style: str | None = None
    consistency_rules: list[str] = Field(default_factory=list)


class ShotPlan(BaseModel):
    """One generated shot with bilingual video-model prompts."""

    shot_id: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0)
    fps: int = Field(default=24, gt=0)
    resolution: str = Field(default="720p", min_length=1)

    action: str | None = None
    location: str | None = None
    characters: list[str] = Field(default_factory=list)
    camera: str | None = None
    lighting: str | None = None

    generation_prompt_en: str | None = None
    generation_prompt_zh: str | None = None

    # V1 compatibility: downstream code may still read shot.prompt.
    prompt: str | None = None

    reference_images: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_prompt_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        result = dict(data)
        prompt = result.get("prompt")
        prompt_en = result.get("generation_prompt_en")
        prompt_zh = result.get("generation_prompt_zh")

        if not any(isinstance(v, str) and v.strip() for v in (prompt, prompt_en, prompt_zh)):
            raise ValueError(
                "ShotPlan requires prompt, generation_prompt_en, or generation_prompt_zh."
            )

        if not prompt:
            result["prompt"] = prompt_en or prompt_zh
        if not prompt_en:
            result["generation_prompt_en"] = result["prompt"]
        if not prompt_zh:
            result["generation_prompt_zh"] = result["prompt"]

        return result


class VideoPlan(BaseModel):
    original_prompt: str = Field(min_length=1)
    target_duration_seconds: float = Field(gt=0)
    detected_language: Literal["en", "zh"] | None = None
    planning_language: Literal["en", "zh"] | None = None

    global_style: str | None = None
    visual_style: VisualStyle | None = None
    character_bible: dict[str, CharacterProfile] = Field(default_factory=dict)
    environment_bible: dict[str, EnvironmentProfile] = Field(default_factory=dict)
    shots: list[ShotPlan] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def planned_duration_seconds(self) -> float:
        return sum(shot.duration_seconds for shot in self.shots)
