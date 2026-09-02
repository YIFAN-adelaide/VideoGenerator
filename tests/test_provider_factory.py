from dataclasses import replace

from app.config import settings as default_settings
from app.providers.factory import build_provider_resources
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
