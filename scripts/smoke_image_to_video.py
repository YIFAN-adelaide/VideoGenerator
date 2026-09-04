from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from app.config import Settings
from app.providers.factory import build_provider_resources
from app.schemas import VideoGenerationRequest
from app.services.image_preprocessor import ImagePreprocessor
from app.services.video_probe import VideoProbe


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one image while preserving its aspect ratio, "
            "then generate one FastWan image-conditioned video."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the original user/reference image.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "The subject begins moving naturally while the "
            "camera remains cinematic and stable."
        ),
    )
    parser.add_argument(
        "--resolution",
        choices=("480p", "720p"),
        default="720p",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    args = parser.parse_args()

    settings = Settings()
    resources = build_provider_resources(
        settings
    )
    provider = resources.provider

    job_id = (
        "image-smoke-"
        + uuid4().hex[:8]
    )

    reference_root = Path(
        "reference_assets"
    ).resolve()

    prepared_path = (
        reference_root
        / "prepared"
        / f"{job_id}.png"
    )

    print("=" * 70)
    print("PREPARE IMAGE")
    print("=" * 70)

    prepared = await ImagePreprocessor().prepare(
        args.image,
        prepared_path,
        resolution=args.resolution,
    )

    print(
        "source:",
        f"{prepared.source_width}x"
        f"{prepared.source_height}",
    )
    print(
        "source aspect:",
        prepared.source_aspect_ratio,
    )
    print(
        "canvas:",
        f"{prepared.canvas_width}x"
        f"{prepared.canvas_height}",
    )
    print(
        "canvas aspect:",
        prepared.canvas_aspect_ratio,
    )
    print(
        "prepared:",
        prepared.output_path,
    )

    request = VideoGenerationRequest(
        prompt=args.prompt,
        duration_seconds=args.duration,
        fps=24,
        resolution=args.resolution,
        seed=args.seed,
        initial_image=str(
            prepared.output_path
        ),
        width=prepared.canvas_width,
        height=prepared.canvas_height,
    )

    print()
    print("=" * 70)
    print("GENERATE")
    print("=" * 70)
    print("job:", job_id)

    result = await provider.generate(
        request,
        job_id,
    )

    probe = await VideoProbe().probe(
        result.output_path
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print("video:", result.output_path)
    print(
        "video size:",
        f"{probe.width}x{probe.height}",
    )
    print("frames:", probe.frame_count)
    print("fps:", probe.fps)
    print(
        "duration:",
        probe.duration_seconds,
    )
    print(
        "input_reference:",
        result.metadata.get(
            "input_reference"
        ),
    )

    close = getattr(
        provider,
        "close",
        None,
    )
    if close is not None:
        await close()


if __name__ == "__main__":
    asyncio.run(main())
