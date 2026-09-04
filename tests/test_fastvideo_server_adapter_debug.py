from __future__ import annotations

import subprocess

import pytest

from tests._diagnostic_env import docker_container_running


CONTAINER = "fastvideo-wan"

FILES = [
    "/FastVideo/fastvideo/entrypoints/openai/protocol.py",
    "/FastVideo/fastvideo/entrypoints/openai/request_adapter.py",
    "/FastVideo/fastvideo/entrypoints/openai/video_api.py",
    "/FastVideo/fastvideo/api/sampling_param.py",
]

PATTERN = (
    "size|width|height|input_reference|reference_url|"
    "num_frames|seconds|Sampling"
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        text=True,
        capture_output=True,
        check=False,
    )


def _docker_exec_shell(command: str) -> subprocess.CompletedProcess[str]:
    return _run(
        "docker",
        "exec",
        CONTAINER,
        "/bin/sh",
        "-lc",
        command,
    )


def test_print_fastvideo_server_request_mapping() -> None:
    """
    AWS/runtime diagnostic. Skips cleanly on local machines without the
    running FastVideo Docker container.
    """
    running, reason = docker_container_running(CONTAINER)
    if not running:
        pytest.skip(
            f"FastVideo Docker runtime unavailable: {reason}"
        )

    print("\n" + "=" * 88)
    print("FASTVIDEO SERVER REQUEST-MAPPING DIAGNOSTIC")
    print("=" * 88)
    print(f"container: {CONTAINER}")

    found_any = False

    for file_path in FILES:
        exists = _docker_exec_shell(
            f'test -f "{file_path}"'
        )

        if exists.returncode != 0:
            print("\n" + "-" * 88)
            print(f"MISSING: {file_path}")
            continue

        found_any = True

        result = _docker_exec_shell(
            "grep -n -C 8 -E "
            f"'{PATTERN}' "
            f'"{file_path}" '
            "|| true"
        )

        print("\n" + "-" * 88)
        print(file_path)
        print("-" * 88)

        text = result.stdout.strip()
        print(text or "(No matching lines found.)")

    assert found_any

    print("\n" + "=" * 88)
    print("SERVER CONFIG")
    print("=" * 88)

    config = _docker_exec_shell(
        'cat /configs/fastwan5b_server.yaml 2>/dev/null || true'
    )

    print(
        config.stdout.strip()
        or "Could not read /configs/fastwan5b_server.yaml"
    )
