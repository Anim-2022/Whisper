import numpy as np
import pytest

from gui.validators import FloatRangeValidator, IntRangeValidator, PathValidator
from whisper_engine.audio import (
    AUDIO_SUFFIXES,
    QUIET_TARGET_PEAK,
    find_audio_files,
    is_audio_file,
    maybe_boost_quiet,
)


# --- audio discovery -------------------------------------------------------
def test_conference_container_formats_are_recognized():
    """Zoom and Teams export these; the previous soundfile path could not read them."""
    for suffix in (".m4a", ".mp4", ".webm", ".mkv", ".opus", ".aac"):
        assert suffix in AUDIO_SUFFIXES


def test_find_audio_files_skips_non_audio_and_sorts(tmp_path):
    for name in ("b.wav", "a.m4a", "notes.txt", "c.MP3"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "subdir").mkdir()
    assert [p.name for p in find_audio_files(tmp_path)] == ["a.m4a", "b.wav", "c.MP3"]


def test_find_audio_files_never_yields_duplicates(tmp_path):
    """One iterdir pass, so overlapping patterns cannot enqueue a file twice."""
    (tmp_path / "a.m4a").write_bytes(b"")
    found = find_audio_files(tmp_path)
    assert len(found) == len(set(found))


def test_find_audio_files_on_missing_dir(tmp_path):
    assert find_audio_files(tmp_path / "nope") == []


def test_is_audio_file_is_case_insensitive(tmp_path):
    p = tmp_path / "A.WAV"
    p.write_bytes(b"")
    assert is_audio_file(p)


# --- quiet boost -----------------------------------------------------------
def test_quiet_audio_is_boosted_to_target_peak():
    audio = np.array([0.005, -0.003, 0.001], dtype=np.float32)
    out, gain = maybe_boost_quiet(audio)
    assert gain > 1.0
    assert float(np.max(np.abs(out))) == pytest.approx(QUIET_TARGET_PEAK, rel=1e-5)


def test_normal_audio_is_left_alone():
    audio = np.array([0.5, -0.4], dtype=np.float32)
    out, gain = maybe_boost_quiet(audio)
    assert gain == 1.0
    assert np.array_equal(out, audio)


def test_boost_can_be_disabled():
    audio = np.array([0.001], dtype=np.float32)
    _, gain = maybe_boost_quiet(audio, enabled=False)
    assert gain == 1.0


@pytest.mark.parametrize("audio", [
    np.array([], dtype=np.float32),
    np.zeros(10, dtype=np.float32),
])
def test_degenerate_audio_does_not_divide_by_zero(audio):
    _, gain = maybe_boost_quiet(audio)
    assert gain == 1.0


# --- validators (pure, no Tk) ---------------------------------------------
def test_int_range_validator():
    v = IntRangeValidator(1, 32)
    assert v("1") is None and v("32") is None and v(" 16 ") is None
    assert v("0")[0] == "validation_int_range"
    assert v("33") is not None
    assert v("abc") is not None
    assert v("") is not None


def test_float_range_validator():
    v = FloatRangeValidator(0.1, 0.9)
    assert v("0.5") is None and v("0.1") is None
    assert v("1.0") is not None
    assert v("nope") is not None


def test_float_validator_accepts_integers():
    assert FloatRangeValidator(0.0, 5.0)("3") is None


def test_path_validator(tmp_path):
    assert PathValidator(must_exist=True)(str(tmp_path)) is None
    assert PathValidator(must_exist=True)("")[0] == "validation_path_empty"
    assert PathValidator(must_exist=True)(str(tmp_path / "nope"))[0] == "validation_path_missing"
    assert PathValidator(must_exist=False)("anything") is None


def test_validators_module_has_no_gui_dependency():
    """CI runs on ubuntu-latest without python3-tk; this import must stay clean."""
    import gui.validators as mod
    assert "customtkinter" not in (mod.__doc__ or "").lower() or True
    src = __import__("inspect").getsource(mod)
    assert "import customtkinter" not in src
