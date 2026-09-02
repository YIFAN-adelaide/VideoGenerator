import pytest

from app.runtime.helios_loader import (
    HeliosModelLoader,
    HeliosRuntimeConfig,
    HeliosRuntimeNotLoadedError,
)


def test_loader_is_lazy_without_gpu_dependencies():
    loader = HeliosModelLoader(HeliosRuntimeConfig())
    assert loader.is_loaded is False
    assert loader.health()["loaded"] is False


def test_get_pipeline_requires_explicit_load():
    loader = HeliosModelLoader(HeliosRuntimeConfig())
    with pytest.raises(HeliosRuntimeNotLoadedError):
        loader.get_pipeline()


def test_compile_and_low_vram_are_mutually_exclusive():
    with pytest.raises(ValueError):
        HeliosRuntimeConfig(low_vram=True, enable_compile=True)


def test_runtime_rejects_cpu():
    with pytest.raises(ValueError):
        HeliosRuntimeConfig(device="cpu")
