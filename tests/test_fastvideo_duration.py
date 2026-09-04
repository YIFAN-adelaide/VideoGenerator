from __future__ import annotations

import pytest

from app.providers.fastvideo_duration import (
    is_fastwan22_ti2v_5b,
    resolve_fastvideo_duration,
)


def test_fastwan_alias_detection():
    assert is_fastwan22_ti2v_5b(
        "fastwan2.2-ti2v-5b"
    )
    assert is_fastwan22_ti2v_5b(
        "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
    )
    assert not is_fastwan22_ti2v_5b(
        "some-other-video-model"
    )


def test_five_seconds_resolves_up_to_121_frames():
    result = resolve_fastvideo_duration(
        model="fastwan2.2-ti2v-5b",
        duration_seconds=5.0,
        fps=24,
    )

    assert result.requested_frames == 120
    assert result.generation_frames == 121
    assert result.generation_seconds == pytest.approx(
        121 / 24
    )
    assert result.uses_explicit_num_frames is True


def test_existing_valid_grid_is_not_changed():
    result = resolve_fastvideo_duration(
        model="fastwan2.2-ti2v-5b",
        duration_seconds=117 / 24,
        fps=24,
    )

    assert result.requested_frames == 117
    assert result.generation_frames == 117


def test_two_seconds_resolves_to_49_frames():
    result = resolve_fastvideo_duration(
        model="fastwan2.2-ti2v-5b",
        duration_seconds=2.0,
        fps=24,
    )

    assert result.requested_frames == 48
    assert result.generation_frames == 49


def test_fastwan_rejects_non_24_fps():
    with pytest.raises(
        ValueError,
        match="requires 24 FPS",
    ):
        resolve_fastvideo_duration(
            model="fastwan2.2-ti2v-5b",
            duration_seconds=5.0,
            fps=30,
        )


def test_unknown_model_keeps_seconds_mode():
    result = resolve_fastvideo_duration(
        model="other-model",
        duration_seconds=5.0,
        fps=24,
    )

    assert result.requested_frames == 120
    assert result.generation_frames is None
    assert result.generation_seconds is None
    assert result.uses_explicit_num_frames is False
