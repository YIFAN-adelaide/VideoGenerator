from __future__ import annotations

from pathlib import Path

from app.providers.fastvideo import FastVideoProvider
from app.schemas import VideoGenerationRequest


def test_image_conditioned_payload_uses_custom_aspect_ratio(
    tmp_path: Path,
) -> None:
    reference_root = (
        tmp_path
        / "reference_assets"
    )
    prepared = (
        reference_root
        / "prepared"
        / "portrait.png"
    )
    prepared.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    prepared.write_bytes(b"png")

    provider = FastVideoProvider(
        base_url="http://fastvideo.test",
        model="fastwan2.2-ti2v-5b",
        output_dir=str(
            tmp_path / "outputs"
        ),
        input_host_dir=reference_root,
        input_container_dir="/inputs",
    )

    request = VideoGenerationRequest(
        prompt="The tiger slowly looks at the camera.",
        duration_seconds=5,
        fps=24,
        resolution="720p",
        seed=123,
        initial_image=str(prepared),
        width=768,
        height=1152,
    )

    payload = provider.build_payload(request)

    assert payload["num_frames"] == 121
    assert payload["fps"] == 24
    assert payload["size"] == "768x1152"
    assert payload["input_reference"] == (
        "/inputs/prepared/portrait.png"
    )
    assert payload["seed"] == 123


def test_text_only_request_keeps_existing_profile_size(
    tmp_path: Path,
) -> None:
    provider = FastVideoProvider(
        base_url="http://fastvideo.test",
        model="fastwan2.2-ti2v-5b",
        output_dir=str(tmp_path),
    )

    request = VideoGenerationRequest(
        prompt="A tiger in snow.",
        duration_seconds=5,
        fps=24,
        resolution="720p",
    )

    payload = provider.build_payload(request)

    assert payload["size"] == "1280x704"
    assert "input_reference" not in payload
