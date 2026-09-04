from __future__ import annotations

import shutil
import subprocess
from textwrap import indent

import pytest


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
    Diagnostic-only test.

    It does NOT generate a video and does NOT use the GPU for inference.
    It prints the relevant source code from the *actual FastVideo container*
    so we can see where OpenAI `/v1/videos` request fields such as:

        size
        width / height
        input_reference
        num_frames

    are translated into FastVideo sampling parameters.

    Run with:

        pytest -q -s tests/test_fastvideo_server_adapter_debug.py
    """
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed in this environment")

    running = _run(
        "docker",
        "inspect",
        "-f",
        "{{.State.Running}}",
        CONTAINER,
    )

    if running.returncode != 0:
        pytest.fail(
            f"Could not inspect Docker container {CONTAINER!r}:\n"
            f"{running.stderr.strip()}"
        )

    if running.stdout.strip().lower() != "true":
        pytest.fail(
            f"Docker container {CONTAINER!r} is not running."
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

        if text:
            print(text)
        else:
            print("(No matching lines found.)")

    assert found_any, (
        "None of the expected FastVideo source files were found "
        "inside the running container."
    )

    # Also print the exact model/server config because a fixed sampling
    # profile in YAML may be overriding the OpenAI request size.
    print("\n" + "=" * 88)
    print("SERVER CONFIG")
    print("=" * 88)

    config = _docker_exec_shell(
        'cat /configs/fastwan5b_server.yaml 2>/dev/null || true'
    )

    if config.stdout.strip():
        print(config.stdout.strip())
    else:
        print(
            "Could not read /configs/fastwan5b_server.yaml "
            "from the container."
        )

    print("\n" + "=" * 88)
    print("WHAT WE ARE LOOKING FOR")
    print("=" * 88)
    print(
        "1. Does request_adapter parse request.size?\n"
        "2. Does it assign parsed values to sampling width/height?\n"
        "3. Does input_reference become the model image input?\n"
        "4. Does fastwan5b_server.yaml force width=832 and height=480?\n"
        "5. Is there a model/profile default applied after request parsing?"
    )
