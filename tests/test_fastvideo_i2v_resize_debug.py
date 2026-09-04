from __future__ import annotations

import shutil
import subprocess

import pytest


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
    Diagnostic-only test.

    It does NOT generate video and does NOT run inference.

    It searches the actual FastVideo container for I2V/reference-image
    preprocessing and resize logic, especially any 480*832/max-area rule.

    Run:

        pytest -q -s tests/test_fastvideo_i2v_resize_debug.py

    The output is intentionally limited to matching filenames and compact
    context blocks so it is easy to paste back into ChatGPT.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")

    running = _run(
        "docker",
        "inspect",
        "-f",
        "{{.State.Running}}",
        CONTAINER,
    )

    if running.returncode != 0:
        pytest.fail(
            f"Could not inspect container {CONTAINER!r}:\n"
            f"{running.stderr.strip()}"
        )

    if running.stdout.strip().lower() != "true":
        pytest.fail(
            f"Container {CONTAINER!r} is not running."
        )

    _print_section("FASTVIDEO I2V RESIZE DIAGNOSTIC")
    print(f"container: {CONTAINER}")

    roots = " ".join(
        f'"{root}"'
        for root in SEARCH_ROOTS
    )

    # First locate likely files using a broad but bounded grep.
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

    if not files:
        print(
            "No likely I2V resize files were found with the first search."
        )

    _print_section("LIKELY SOURCE FILES")
    for file_path in files:
        print(file_path)

    # Search for the most suspicious rules first.
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

    # Then inspect compact context around image conditioning / resizing.
    compact_pattern = (
        "input_reference|reference_image|"
        "resize|image_size|height|width"
    )

    for file_path in files[:12]:
        result = _docker_shell(
            "grep -n -C 8 -E "
            f"'{compact_pattern}' "
            f'"{file_path}" 2>/dev/null | head -n 140'
        )

        if result.stdout.strip():
            _print_section(
                f"CONTEXT: {file_path}"
            )
            print(result.stdout.strip())

    # Show relevant package versions because behavior can differ by release.
    _print_section("INSTALLED FASTVIDEO VERSION")
    version_result = _docker_shell(
        "/opt/venv/bin/python - <<'PY'\n"
        "try:\n"
        "    import importlib.metadata as m\n"
        "    print('fastvideo:', m.version('fastvideo'))\n"
        "except Exception as exc:\n"
        "    print('fastvideo version unavailable:', exc)\n"
        "try:\n"
        "    import diffusers\n"
        "    print('diffusers:', diffusers.__version__)\n"
        "except Exception as exc:\n"
        "    print('diffusers version unavailable:', exc)\n"
        "PY"
    )
    print(version_result.stdout.strip())

    _print_section("INTERPRETATION GUIDE")
    print(
        "What matters most:\n"
        "1. Any code that computes a max area around 480*832.\n"
        "2. Any code that resizes input_reference before sampling.\n"
        "3. Any code that replaces request width/height after request_adapter.\n"
        "4. Whether the resize preserves aspect ratio or forces 832x480.\n"
        "5. Whether the TI2V pipeline has its own default dimensions.\n"
    )

    assert files or found_suspicious, (
        "The diagnostic could not find any likely image/I2V source files."
    )
