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
    """
    explicit = os.getenv("DEBUG_INITIAL_IMAGE")

    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            pytest.fail(
                f"DEBUG_INITIAL_IMAGE does not exist: {path}"
            )
        return path

    prepared_dir = Path("reference_assets/prepared").resolve()

    if not prepared_dir.exists():
        pytest.fail(
            f"Prepared reference directory does not exist: {prepared_dir}"
        )

    candidates = sorted(
        (
            p for p in prepared_dir.glob("*.png")
            if p.is_file()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        pytest.fail(
            f"No prepared PNG images found in {prepared_dir}"
        )

    return candidates[0]


@pytest.mark.asyncio
async def test_print_real_fastvideo_payload() -> None:
    """
    Diagnostic-only test.

    It DOES NOT call FastVideo and DOES NOT use the GPU.
    It only shows exactly what VideoGenerator's FastVideoProvider would send
    to POST /v1/videos for the newest prepared reference image.
    """
    settings = Settings()
    resources = build_provider_resources(settings)
    provider = resources.provider

    image_path = _find_reference_image()
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

        assert payload.get("input_reference"), (
            "input_reference is missing from the FastVideo payload"
        )

        expected_size = (
            f"{image_info.width}x{image_info.height}"
        )
        assert payload.get("size") == expected_size, (
            "FastVideo payload size does not match the prepared "
            f"reference image. Expected {expected_size}, "
            f"got {payload.get('size')!r}"
        )

        assert payload.get("num_frames") == 121, (
            "Expected the FastWan 5-second request to use 121 frames"
        )

    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            await close()
