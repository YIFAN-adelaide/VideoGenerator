from types import SimpleNamespace

import pytest

from app.providers.helios import HeliosProvider
from app.schemas import VideoGenerationRequest


class FakeLoader:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = SimpleNamespace(device="cuda:0")

    def get_pipeline(self):
        return self.pipeline


class FakePipeline:
    def __init__(self):
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            frames=[["frame-1", "frame-2"]]
        )


def fake_exporter(frames, output_path, fps):
    with open(output_path, "wb") as f:
        f.write(b"fake-mp4")


@pytest.mark.asyncio
async def test_provider_builds_params_runs_pipeline_and_exports(tmp_path):
    pipeline = FakePipeline()
    loader = FakeLoader(pipeline)

    provider = HeliosProvider(
        loader=loader,
        output_dir=str(tmp_path),
        exporter=fake_exporter,
        generator_factory=lambda seed: f"generator:{seed}",
    )

    request = VideoGenerationRequest(
        prompt="A tiger walking through snow",
        duration_seconds=4,
        fps=24,
        resolution="480p",
        seed=123,
    )

    result = await provider.generate(request, "job-123")

    assert result.output_path.endswith("job-123.mp4")
    assert result.metadata["provider"] == "helios"
    assert result.metadata["num_frames"] == 99
    assert result.metadata["seed"] == 123
    assert result.metadata["width"] == 640
    assert result.metadata["height"] == 384

    assert pipeline.last_kwargs["num_frames"] == 99
    assert pipeline.last_kwargs["guidance_scale"] == 1.0
    assert pipeline.last_kwargs["generator"] == "generator:123"
    assert pipeline.last_kwargs[
        "pyramid_num_inference_steps_list"
    ] == [2, 2, 2]


def test_provider_rejects_experimental_720p_by_default(tmp_path):
    provider = HeliosProvider(
        loader=FakeLoader(FakePipeline()),
        output_dir=str(tmp_path),
        exporter=fake_exporter,
        generator_factory=lambda seed: seed,
    )

    request = VideoGenerationRequest(
        prompt="test",
        duration_seconds=4,
        fps=24,
        resolution="720p",
        seed=1,
    )

    with pytest.raises(ValueError):
        provider.build_params(request)
