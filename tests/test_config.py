from pathlib import Path

import pytest

from whisper_engine.config import TranscriptionConfig


def cfg(**kw):
    """Config pinned to CPU so results do not depend on the test machine's GPU."""
    kw.setdefault("device", "cpu")
    return TranscriptionConfig(**kw)


# --- validation ------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    {"beam_size": 0},
    {"batch_size": 0},
    {"vad_threshold": 0.0},
    {"vad_threshold": 1.0},
    {"vad_min_silence_ms": -1},
    {"compute_type": "float8"},
    {"max_new_tokens": 0},
])
def test_invalid_values_rejected(kw):
    with pytest.raises(ValueError):
        cfg(**kw)


def test_cpu_never_uses_float16():
    """float16 on CPU is slow and numerically poor for Whisper."""
    assert cfg(device="cpu", compute_type="float16").compute_type == "int8"
    assert cfg(device="cpu", compute_type="auto").compute_type == "int8"


def test_formats_are_normalized():
    c = cfg(formats=(".TXT", "srt", "txt"))
    assert c.formats == ("txt", "srt"), "lowercased, dots stripped, deduplicated, order kept"


def test_formats_never_end_up_empty():
    assert cfg(formats=()).formats == ("txt",)
    assert cfg(formats="md").formats == ("md",)


def test_output_dir_accepts_a_string():
    assert isinstance(cfg(output_dir="./out").output_dir, Path)


# --- the two silent-failure invariants ------------------------------------
def test_hallucination_filter_forces_word_timestamps():
    """faster-whisper only applies the filter inside the word-timestamp path."""
    c = cfg(batched=False, hallucination_silence_threshold=2.0, word_timestamps=False)
    assert c.word_timestamps is True
    assert any("word timestamps" in n for n in c.notes)


def test_batched_mode_clears_options_it_would_discard():
    """BatchedInferencePipeline drops these; the config must not claim otherwise."""
    c = cfg(batched=True, condition_on_previous_text=True,
            hallucination_silence_threshold=2.0, temperature_fallback=True)
    assert c.condition_on_previous_text is False
    assert c.hallucination_silence_threshold is None
    assert c.temperature_fallback is False
    assert any("batched mode ignores" in n for n in c.notes)


def test_batched_without_vad_falls_back_to_sequential():
    """The batched pipeline derives clip timestamps from VAD and raises without it.

    Turning VAD back on would override an explicit choice, so the mode gives way
    instead. Before this, unticking "Enable VAD" in batched mode crashed the run
    with "No clip timestamps found".
    """
    c = cfg(batched=True, vad_filter=False)
    assert c.batched is False
    assert any("requires VAD" in n for n in c.notes)


def test_batched_with_vad_stays_batched():
    assert cfg(batched=True, vad_filter=True).batched is True


def test_batched_warns_when_subtitles_are_requested():
    """A 30 s cue is a fine transcript line and a useless subtitle."""
    c = cfg(batched=True, formats=("txt", "srt"))
    assert any("subtitle cues" in n for n in c.notes)
    assert not any("subtitle cues" in n for n in cfg(batched=True, formats=("txt",)).notes)
    assert not any("subtitle cues" in n for n in cfg(batched=False, formats=("srt",)).notes)


def test_sequential_mode_keeps_those_options():
    c = cfg(batched=False, condition_on_previous_text=True,
            hallucination_silence_threshold=2.0, temperature_fallback=True)
    assert c.condition_on_previous_text is True
    assert c.hallucination_silence_threshold == 2.0
    assert c.temperature_fallback is True


# --- derived properties ----------------------------------------------------
def test_temperature_ladder_is_used_only_with_fallback():
    assert cfg(batched=False, temperature_fallback=True).temperature[0] == 0.0
    assert cfg(batched=False, temperature_fallback=False).temperature == 0.0


def test_language_is_none_when_auto():
    assert cfg(auto_lang=True).language is None
    assert cfg(auto_lang=False, lang="de").language == "de"


def test_options_dict_reflects_the_effective_run():
    """Metadata must describe what actually ran, not what was requested."""
    opts = cfg(batched=True, condition_on_previous_text=True).to_options_dict()
    assert opts["condition_on_previous_text"] is False
    assert opts["batch_size"] is not None
    assert cfg(batched=False).to_options_dict()["batch_size"] is None


def test_describe_mentions_model_and_mode():
    text = cfg(model_id="whisper-medium", batched=True).describe()
    assert "whisper-medium" in text and "batched" in text
