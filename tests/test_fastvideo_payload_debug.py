from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from app.config import Settings
from app.providers.factory import build_provider_resources
from app.schemas import VideoGenerationRequest
from app.services.image_probe import ImageProbe


def _find_reference_image() -> Path:
    """
    Use DEBUG_INITIAL_IMAGE when supplied; otherwise use the newest prepared
    reference image created by scripts/smoke_image_to_video.py.

    This is an environment diagnostic, so lack of an AWS/prepared asset causes
    a pytest skip rather than a local-suite failure.
    """
    explicit = os.getenv("DEBUG_INITIAL_IMAGE")

    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            pytest.skip(
                f"DEBUG_INITIAL_IMAGE does not exist: {path}"
            )
        return path

    prepared_dir = Path("reference_assets/prepared").resolve()

    if not prepared_dir.exists():
        pytest.skip(
            "No prepared reference_assets directory in this environment. "
            "Run this diagnostic on AWS after an image-to-video preparation "
            "step, or set DEBUG_INITIAL_IMAGE."
        )

    candidates = sorted(
        (
            p
            for p in prepared_dir.glob("*.png")
            if p.is_file()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        pytest.skip(
            f"No prepared PNG images found in {prepared_dir}"
        )

    return candidates[0]


@pytest.mark.asyncio
async def test_print_real_fastvideo_payload() -> None:
    """
    Diagnostic-only test.

    It DOES NOT call FastVideo and DOES NOT use the GPU.
    It prints exactly what FastVideoProvider would send to /v1/videos.
    """
    image_path = _find_reference_image()

    settings = Settings()
    resources = build_provider_resources(settings)
    provider = resources.provider

    if not hasattr(provider, "build_payload"):
        pytest.skip(
            "Current environment is not configured with FastVideoProvider. "
            "Load the FastVideo .env or run this diagnostic on AWS."
        )

    image_info = await ImageProbe().probe(image_path)

    request = VideoGenerationRequest(
        prompt=(
            "The tiger continues walking naturally "
            "through the snowy forest."
        ),
        duration_seconds=5,
        fps=24,
        resolution="720p",
        seed=123,
        initial_image=str(image_path),
        width=image_info.width,
        height=image_info.height,
    )

    try:
        payload = provider.build_payload(request)

        print("\n" + "=" * 72)
        print("FASTVIDEO PAYLOAD DIAGNOSTIC")
        print("=" * 72)
        print("provider class :", type(provider).__name__)
        print("model          :", getattr(provider, "model", None))
        print("source image   :", image_path)
        print(
            "image size     :",
            f"{image_info.width}x{image_info.height}",
        )
        print("-" * 72)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("=" * 72)

        assert payload.get("input_reference")

        expected_size = f"{image_info.width}x{image_info.height}"
        assert payload.get("size") == expected_size

        assert payload.get("num_frames") == 121

    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            await close()
