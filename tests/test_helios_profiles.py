import pytest

from app.providers.helios_profiles import get_helios_profile


def test_standard_profile_matches_initial_helios_resolution():
    profile = get_helios_profile("480p")

    assert profile.width == 640
    assert profile.height == 384
    assert profile.experimental is False


def test_720p_is_disabled_by_default():
    with pytest.raises(ValueError):
        get_helios_profile("720p")


def test_720p_can_be_explicitly_enabled():
    profile = get_helios_profile(
        "720p",
        allow_experimental=True,
    )

    assert profile.width == 1280
    assert profile.height == 720
    assert profile.experimental is True
