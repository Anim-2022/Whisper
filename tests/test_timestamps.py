import pytest

from whisper_engine.timestamps import format_clock, format_duration, format_timestamp


@pytest.mark.parametrize("seconds,expected", [
    (0.0, "00:00:00,000"),
    (1.5, "00:00:01,500"),
    (61.25, "00:01:01,250"),
    (3661.001, "01:01:01,001"),
    (7322.5, "02:02:02,500"),
])
def test_srt_basic(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_millisecond_carry():
    """round((3.9996 - 3) * 1000) == 1000, which must roll into the seconds field.

    Without the carry branch this renders as the invalid "00:00:03,1000".
    """
    assert format_timestamp(3.9996) == "00:00:04,000"


def test_carry_rolls_through_minute_boundary():
    assert format_timestamp(59.9999) == "00:01:00,000"


def test_carry_rolls_through_hour_boundary():
    assert format_timestamp(3599.9999) == "01:00:00,000"


@pytest.mark.parametrize("bad", [None, -1.0, -0.0001])
def test_negative_and_none_clamp_to_zero(bad):
    assert format_timestamp(bad) == "00:00:00,000"


def test_vtt_separator_is_a_dot():
    assert format_timestamp(61.25, sep=".") == "00:01:01.250"


def test_hours_always_present_by_default():
    """SRT requires the hours field even for short media; fw's own helper omits it."""
    assert format_timestamp(5.0).startswith("00:00:05")
    assert format_timestamp(5.0, always_hours=False) == "00:05,000"


@pytest.mark.parametrize("seconds,expected", [
    (0, "0:00:00"), (65, "0:01:05"), (3600, "1:00:00"), (10289, "2:51:29"),
])
def test_format_clock(seconds, expected):
    assert format_clock(seconds) == expected


@pytest.mark.parametrize("seconds,expected", [
    (5, "5s"), (65, "1m 05s"), (3600, "1h 00m"), (10289, "2h 51m"),
])
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected
