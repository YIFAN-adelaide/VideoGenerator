from types import SimpleNamespace

import pytest

import app.main as main_module


class FakeLoader:
    def __init__(self):
        self.loaded = False
        self.load_calls = 0
        self.unload_calls = 0

    def load(self):
        self.loaded = True
        self.load_calls += 1

    def unload(self):
        self.loaded = False
        self.unload_calls += 1


@pytest.mark.asyncio
async def test_lifespan_loads_and_unloads_helios_runtime(monkeypatch):
    fake_loader = FakeLoader()
    fake_resources = SimpleNamespace(
        helios_loader=fake_loader,
    )

    monkeypatch.setattr(
        main_module,
        "resources",
        fake_resources,
    )

    async with main_module.lifespan(main_module.app):
        assert fake_loader.loaded is True
        assert fake_loader.load_calls == 1

    assert fake_loader.loaded is False
    assert fake_loader.unload_calls == 1
