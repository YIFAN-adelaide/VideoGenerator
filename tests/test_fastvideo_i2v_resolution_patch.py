from __future__ import annotations

import subprocess

import pytest

from tests._diagnostic_env import docker_container_running


CONTAINER = "fastvideo-wan"
TARGET = (
    "/FastVideo/fastvideo/pipelines/stages/"
    "input_validation.py"
)

OLD_LINE = "max_area = 480 * 832"
NEW_LINE = "max_area = batch.height * batch.width"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        text=True,
        capture_output=True,
        check=False,
    )


def test_running_fastvideo_has_requested_canvas_patch() -> None:
    """
    AWS/runtime diagnostic.

    Skips cleanly on local machines where Docker Desktop / the FastVideo
    container is unavailable.
    """
    running, reason = docker_container_running(CONTAINER)
    if not running:
        pytest.skip(
            f"FastVideo Docker runtime unavailable: {reason}"
        )

    result = _run(
        "docker",
        "exec",
        CONTAINER,
        "/bin/sh",
        "-lc",
        (
            f"grep -n -C 5 'max_area' '{TARGET}' "
            "2>/dev/null || true"
        ),
    )

    assert result.returncode == 0

    output = result.stdout

    print("\n" + "=" * 80)
    print("RUNNING FASTVIDEO WAN I2V RESOLUTION POLICY")
    print("=" * 80)
    print(output.strip())
    print("=" * 80)

    assert NEW_LINE in output
    assert OLD_LINE not in output
