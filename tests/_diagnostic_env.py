from __future__ import annotations

import shutil
import subprocess


def docker_container_running(container: str) -> tuple[bool, str]:
    """
    Return (running, reason).

    This is intentionally local-safe: a missing Docker CLI, stopped Docker
    Desktop, unavailable daemon, or missing container is treated as
    "environment unavailable" rather than a test failure.
    """
    if shutil.which("docker") is None:
        return False, "docker CLI is not installed"

    result = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Running}}",
            container,
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        reason = (
            result.stderr.strip()
            or result.stdout.strip()
            or "docker inspect failed"
        )
        return False, reason

    if result.stdout.strip().lower() != "true":
        return False, f"container {container!r} is not running"

    return True, ""
