from __future__ import annotations

from app.providers.fastvideo import FastVideoProvider
from app.schemas import VideoGenerationRequest


def test_fastwan_payload_uses_121_explicit_frames(tmp_path):
    provider = FastVideoProvider(
        base_url="http://fastvideo.test",
        model="fastwan2.2-ti2v-5b",
        output_dir=str(tmp_path),
    )

    request = VideoGenerationRequest(
        prompt="A tiger walking through snow",
        duration_seconds=5,
        fps=24,
        resolution="720p",
        seed=123,
    )

    payload = provider.build_payload(request)

    assert payload["num_frames"] == 121
    assert "seconds" not in payload
    assert payload["fps"] == 24
    assert payload["size"] == "1280x704"
    assert payload["seed"] == 123


def test_unknown_model_payload_preserves_seconds(tmp_path):
    provider = FastVideoProvider(
        base_url="http://fastvideo.test",
        model="other-model",
        output_dir=str(tmp_path),
    )

    request = VideoGenerationRequest(
        prompt="test",
        duration_seconds=5,
        fps=24,
        resolution="720p",
    )

    payload = provider.build_payload(request)

    assert payload["seconds"] == 5.0
    assert "num_frames" not in payload
