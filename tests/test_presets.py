"""Presets must translate into a valid config without any silent surprises."""
from __future__ import annotations

import pytest

from gui import constants as C
from whisper_engine.config import TranscriptionConfig


@pytest.mark.parametrize("name", list(C.PRESETS))
@pytest.mark.parametrize("has_cuda", [True, False])
def test_preset_builds_a_valid_config(name, has_cuda):
    kwargs = C.preset_to_config_kwargs(C.PRESETS[name], has_cuda=has_cuda)
    cfg = TranscriptionConfig(device="cpu", model_id=C.DEFAULT_MODEL, **kwargs)
    assert cfg.beam_size >= 1
    assert cfg.batch_size >= 1
    assert cfg.formats


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_no_preset_asks_for_options_it_would_not_get(name):
    """A preset that trips the batched-mode warning is misconfigured.

    Combining `batched` with condition-on-previous-text or the hallucination
    filter would make the UI advertise settings faster-whisper discards.
    """
    kwargs = C.preset_to_config_kwargs(C.PRESETS[name], has_cuda=True)
    cfg = TranscriptionConfig(device="cpu", **kwargs)
    assert not any("batched mode ignores" in note for note in cfg.notes), cfg.notes


def test_cpu_gets_a_smaller_batch_than_cuda():
    for preset in C.PRESETS.values():
        assert (C.preset_to_config_kwargs(preset, has_cuda=False)["batch_size"]
                <= C.preset_to_config_kwargs(preset, has_cuda=True)["batch_size"])


def test_default_preset_is_the_sequential_quality_one():
    """The benchmark's best-scoring configuration is what new users get."""
    default = C.PRESETS[C.DEFAULT_PRESET]
    assert C.DEFAULT_PRESET in C.PRESETS
    assert default["batched"] is False
    assert default["beam_size"] == 5


@pytest.mark.parametrize("name", ["meeting", "difficult"])
def test_sequential_presets_use_the_options_only_they_can(name):
    """Sequential is only worth its cost if it actually uses the extra guards."""
    preset = C.PRESETS[name]
    assert preset["batched"] is False
    assert preset["temperature_fallback"] is True
    assert preset["hallucination_silence_threshold"] is not None


def test_previous_text_context_is_off_everywhere():
    """Measured: faster-whisper's default of True collapsed punctuation from
    24 marks per 1000 chars to 2 on the Russian lecture."""
    for name, preset in C.PRESETS.items():
        assert preset["condition_on_previous_text"] is False, name


def test_no_preset_disables_vad():
    """Measured: VAD off cost roughly half the accuracy and all punctuation."""
    for name, preset in C.PRESETS.items():
        assert preset["vad_enabled"] is True, name


def test_every_preset_uses_a_wide_beam():
    """Beam 1 saves 12-16 % and costs accuracy on hard audio, so no preset takes it."""
    for name, preset in C.PRESETS.items():
        assert preset["beam_size"] == 5, name


def test_difficult_differs_from_the_default_only_where_it_measurably_helped():
    """A one-variable sweep on degraded audio found exactly two useful changes.

    The decoding thresholds produced byte-identical output at every value tried,
    and extra speech padding was actively harmful, so this preset must not drift
    into changing them for the look of it.
    """
    meeting, difficult = C.PRESETS["meeting"], C.PRESETS["difficult"]
    differing = {k for k in meeting if meeting[k] != difficult[k]}
    assert differing == {"vad_threshold", "vad_min_speech_ms"}, differing
    assert difficult["vad_threshold"] < meeting["vad_threshold"]
    assert difficult["vad_min_speech_ms"] < meeting["vad_min_speech_ms"]


def test_long_preset_is_the_only_batched_one():
    batched = [n for n, p in C.PRESETS.items() if p["batched"]]
    assert batched == ["long"]


def test_defaults_mirror_the_default_preset():
    for key, value in C.PRESETS[C.DEFAULT_PRESET].items():
        if key in ("batch_size_cuda", "batch_size_cpu"):
            continue
        assert C.DEFAULTS[key] == value


def test_every_preset_has_the_full_key_set():
    """A missing key would silently fall back to a config default."""
    reference = set(C.PRESETS[C.DEFAULT_PRESET])
    for name, preset in C.PRESETS.items():
        assert set(preset) == reference, f"{name} differs: {set(preset) ^ reference}"
