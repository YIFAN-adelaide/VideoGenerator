import pytest

from app.providers.helios_parameters import (
    HeliosGenerationParams,
    normalize_helios_frame_count,
)


def test_four_second_request_rounds_to_99_frames():
    requested, normalized, actual = normalize_helios_frame_count(
        duration_seconds=4,
        fps=24,
    )

    assert requested == 96
    assert normalized == 99
    assert actual == pytest.approx(4.125)


def test_ten_second_request_rounds_to_264_frames():
    requested, normalized, actual = normalize_helios_frame_count(
        duration_seconds=10,
        fps=24,
    )

    assert requested == 240
    assert normalized == 264
    assert actual == pytest.approx(11.0)


def test_generation_params_reject_non_33_multiple():
    with pytest.raises(ValueError):
        HeliosGenerationParams(
            prompt="test",
            profile_name="test",
            requested_resolution="480p",
            width=640,
            height=384,
            requested_duration_seconds=4,
            fps=24,
            requested_frames=96,
            num_frames=96,
            actual_duration_seconds=4,
            seed=42,
            guidance_scale=1.0,
            pyramid_num_inference_steps_list=(2, 2, 2),
        )
