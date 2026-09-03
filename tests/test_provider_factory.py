from dataclasses import replace

from app.config import settings as default_settings
from app.providers.factory import build_provider_resources
from app.providers.fastvideo import FastVideoProvider
from app.providers.helios import HeliosProvider
from app.providers.mock import MockVideoProvider


def test_factory_builds_mock_provider_without_gpu_runtime(tmp_path):
    settings = replace(
        default_settings,
        video_provider="mock",
        output_dir=str(tmp_path),
    )

    resources = build_provider_resources(settings)

    assert isinstance(resources.provider, MockVideoProvider)
    assert resources.helios_loader is None


def test_factory_builds_helios_lazily_without_loading_cuda(tmp_path):
    settings = replace(
        default_settings,
        video_provider="helios",
        output_dir=str(tmp_path),
        helios_repo_path=None,
        helios_base_model_path="BestWishYsh/Helios-Distilled",
    )

    resources = build_provider_resources(settings)

    assert isinstance(resources.provider, HeliosProvider)
    assert resources.helios_loader is not None
    assert resources.helios_loader.is_loaded is False


def test_factory_builds_fastvideo_external_provider(tmp_path):
    settings = replace(
        default_settings,
        video_provider="fastvideo",
        output_dir=str(tmp_path),
        fastvideo_base_url="http://127.0.0.1:9200",
        fastvideo_model=(
            "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
        ),
    )

    resources = build_provider_resources(settings)

    assert isinstance(resources.provider, FastVideoProvider)
    assert resources.helios_loader is None
    assert resources.provider.model.endswith(
        "FastWan2.2-TI2V-5B-FullAttn-Diffusers"
    )
