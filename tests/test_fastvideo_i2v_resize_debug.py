from __future__ import annotations

import subprocess

import pytest

from tests._diagnostic_env import docker_container_running


CONTAINER = "fastvideo-wan"

SEARCH_ROOTS = [
    "/FastVideo/fastvideo",
    "/FastVideo/examples",
]

PATTERNS = [
    "480[[:space:]]*\\*[[:space:]]*832",
    "832[[:space:]]*\\*[[:space:]]*480",
    "max_area",
    "max_pixels",
    "input_reference",
    "reference_image",
    "image_size",
    "resize",
    "height",
    "width",
]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        text=True,
        capture_output=True,
        check=False,
    )


def _docker_shell(command: str) -> subprocess.CompletedProcess[str]:
    return _run(
        "docker",
        "exec",
        CONTAINER,
        "/bin/sh",
        "-lc",
        command,
    )


def _print_section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def test_locate_fastvideo_i2v_resize_logic() -> None:
    """
    AWS/runtime diagnostic. Skips cleanly on local machines without the
    running FastVideo Docker container.
    """
    running, reason = docker_container_running(CONTAINER)
    if not running:
        pytest.skip(
            f"FastVideo Docker runtime unavailable: {reason}"
        )

    _print_section("FASTVIDEO I2V RESIZE DIAGNOSTIC")
    print(f"container: {CONTAINER}")

    roots = " ".join(
        f'"{root}"'
        for root in SEARCH_ROOTS
    )

    pattern_expr = "|".join(PATTERNS)

    locate_cmd = (
        f"grep -RIlE '{pattern_expr}' {roots} "
        "2>/dev/null | "
        "grep -Ei "
        "'wan|i2v|image|reference|request|sampling|pipeline|processor' "
        "| head -n 40"
    )

    located = _docker_shell(locate_cmd)

    files = [
        line.strip()
        for line in located.stdout.splitlines()
        if line.strip()
    ]

    _print_section("LIKELY SOURCE FILES")
    for file_path in files:
        print(file_path)

    suspicious_patterns = [
        "480[[:space:]]*\\*[[:space:]]*832",
        "832[[:space:]]*\\*[[:space:]]*480",
        "max_area",
        "max_pixels",
    ]

    found_suspicious = False

    for pattern in suspicious_patterns:
        result = _docker_shell(
            f"grep -RIn -C 10 -E '{pattern}' {roots} "
            "2>/dev/null | head -n 160"
        )

        if result.stdout.strip():
            found_suspicious = True
            _print_section(
                f"SUSPICIOUS MATCH: {pattern}"
            )
            print(result.stdout.strip())

    assert files or found_suspicious
