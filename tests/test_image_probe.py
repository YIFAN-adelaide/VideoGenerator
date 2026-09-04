from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.image_probe import (
    ImageProbe,
    ImageProbeError,
)


class FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(
        self,
    ) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_probe_reads_image_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "portrait.png"
    image.write_bytes(b"fake-image")

    payload = {
        "streams": [
            {
                "width": 1024,
                "height": 1536,
                "pix_fmt": "rgb24",
            }
        ]
    }

    async def fake_create_subprocess_exec(
        *args,
        **kwargs,
    ):
        return FakeProcess(
            stdout=json.dumps(
                payload
            ).encode("utf-8")
        )

    monkeypatch.setattr(
        (
            "app.services.image_probe."
            "asyncio.create_subprocess_exec"
        ),
        fake_create_subprocess_exec,
    )

    result = await ImageProbe().probe(image)

    assert result.width == 1024
    assert result.height == 1536
    assert result.aspect_ratio == pytest.approx(
        2 / 3
    )
    assert result.pixel_format == "rgb24"


@pytest.mark.asyncio
async def test_probe_rejects_missing_image(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ImageProbeError,
        match="does not exist",
    ):
        await ImageProbe().probe(
            tmp_path / "missing.png"
        )
