from __future__ import annotations

import math
import re

from app.director.base import BaseDirector
from app.director.video_plan import DirectorRequest, ShotPlan, VideoPlan, VisualStyle

_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _detect_language(text: str) -> str:
    return "zh" if _CJK_PATTERN.search(text) else "en"


class MockDirector(BaseDirector):
    """Deterministic bilingual Director used before the vLLM service is deployed."""

    def __init__(self) -> None:
        super().__init__(director_name="mock", model_name="mock-director-bilingual-v2")

    async def create_plan(self, request: DirectorRequest) -> VideoPlan:
        detected_language = (
            _detect_language(request.prompt)
            if request.user_language == "auto"
            else request.user_language
        )
        planning_language = (
            detected_language
            if request.planning_language == "auto"
            else request.planning_language
        )

        max_duration = request.max_shot_duration_seconds
        num_shots = math.ceil(request.target_duration_seconds / max_duration)
        remaining = request.target_duration_seconds
        shots: list[ShotPlan] = []

        cameras_en = (
            "wide cinematic establishing shot",
            "medium tracking shot",
            "close cinematic shot",
            "slow controlled dolly shot",
        )
        cameras_zh = (
            "电影感广角建立镜头",
            "中景跟拍镜头",
            "电影感近景镜头",
            "缓慢稳定的推轨镜头",
        )

        for index in range(num_shots):
            duration = min(max_duration, remaining)
            shot_number = index + 1
            camera_en = cameras_en[index % len(cameras_en)]
            camera_zh = cameras_zh[index % len(cameras_zh)]

            prompt_en = (
                f"Create cinematic shot {shot_number} of {num_shots} based on this "
                f"story request: {request.prompt} Maintain the same subjects, environment, "
                f"visual identity, lighting, and cinematic style across all shots. "
                f"Camera: {camera_en}."
            )
            prompt_zh = (
                f"根据以下故事要求生成第 {shot_number}/{num_shots} 个电影感镜头：{request.prompt}。"
                f"所有镜头必须保持角色、环境、视觉身份、光照和电影风格一致。"
                f"镜头语言：{camera_zh}。"
            )

            shots.append(
                ShotPlan(
                    shot_id=f"shot_{shot_number:03d}",
                    duration_seconds=duration,
                    fps=request.fps,
                    resolution=request.resolution,
                    prompt=prompt_zh if detected_language == "zh" else prompt_en,
                    generation_prompt_en=prompt_en,
                    generation_prompt_zh=prompt_zh,
                    action=(
                        f"Continue the requested story during shot {shot_number}."
                        if planning_language == "en"
                        else f"在第 {shot_number} 个镜头中继续推进故事。"
                    ),
                    camera=camera_en if planning_language == "en" else camera_zh,
                )
            )
            remaining -= duration

        return VideoPlan(
            original_prompt=request.prompt,
            target_duration_seconds=request.target_duration_seconds,
            detected_language=detected_language,
            planning_language=planning_language,
            global_style=(
                "cinematic, photorealistic, visually consistent"
                if planning_language == "en"
                else "电影感、写实、视觉风格一致"
            ),
            visual_style=VisualStyle(
                look=("cinematic photorealism" if planning_language == "en" else "电影感写实风格"),
                camera_motion=(
                    "controlled cinematic movement"
                    if planning_language == "en"
                    else "稳定、受控的电影镜头运动"
                ),
                consistency_rules=[
                    (
                        "Keep the same subject identity across every shot."
                        if planning_language == "en"
                        else "所有镜头保持同一主体身份。"
                    )
                ],
            ),
            shots=shots,
            metadata={
                "director": self.director_name,
                "model": self.model_name,
                "bilingual_prompts": True,
            },
        )
