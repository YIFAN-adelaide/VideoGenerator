from __future__ import annotations

import importlib.metadata
import py_compile
from pathlib import Path


TARGET = Path(
    "/FastVideo/fastvideo/pipelines/stages/input_validation.py"
)

EXPECTED_VERSION = "0.2.1"

OLD_LINE = "max_area = 480 * 832"
NEW_LINE = "max_area = batch.height * batch.width"

REQUIRED_CONTEXT = (
    "best_output_size(iw, ih, dw, dh, max_area)"
)


def main() -> None:
    try:
        installed_version = importlib.metadata.version("fastvideo")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "FastVideo package metadata was not found. "
            "Refusing to patch an unknown image."
        ) from exc

    print(f"FastVideo version: {installed_version}")

    if installed_version != EXPECTED_VERSION:
        raise RuntimeError(
            "FastVideo version mismatch. "
            f"Expected {EXPECTED_VERSION}, got {installed_version}. "
            "Review the upstream input-validation implementation before "
            "updating this patch."
        )

    if not TARGET.exists():
        raise RuntimeError(
            f"Expected FastVideo source file does not exist: {TARGET}"
        )

    source = TARGET.read_text(encoding="utf-8")

    if REQUIRED_CONTEXT not in source:
        raise RuntimeError(
            "Expected best_output_size() call was not found. "
            "The upstream implementation may have changed."
        )

    old_count = source.count(OLD_LINE)
    new_count = source.count(NEW_LINE)

    if old_count == 0 and new_count == 1:
        print("Resolution patch is already applied.")
        return

    if old_count != 1:
        raise RuntimeError(
            "Refusing to patch because the expected hard-coded Wan I2V "
            f"resolution rule occurred {old_count} times instead of exactly 1."
        )

    if new_count != 0:
        raise RuntimeError(
            "Refusing to patch because the replacement rule already appears "
            "in an unexpected state."
        )

    patched = source.replace(
        OLD_LINE,
        (
            "# Respect the generation canvas requested by VideoGenerator.\n"
            "                # best_output_size() still aligns the image to the "
            "Wan/VAE grid\n"
            "                # while preserving the input-reference aspect ratio.\n"
            "                max_area = batch.height * batch.width"
        ),
        1,
    )

    TARGET.write_text(
        patched,
        encoding="utf-8",
    )

    verification = TARGET.read_text(encoding="utf-8")

    if OLD_LINE in verification:
        raise RuntimeError(
            "Patch verification failed: old max_area rule still exists."
        )

    if verification.count(NEW_LINE) != 1:
        raise RuntimeError(
            "Patch verification failed: replacement max_area rule is missing "
            "or duplicated."
        )

    py_compile.compile(
        str(TARGET),
        doraise=True,
    )

    print("Wan I2V resolution policy patched successfully.")
    print(f"Target: {TARGET}")
    print(f"Old: {OLD_LINE}")
    print(f"New: {NEW_LINE}")


if __name__ == "__main__":
    main()
