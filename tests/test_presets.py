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


def test_default_preset_exists_and_is_batched():
    assert C.DEFAULT_PRESET in C.PRESETS
    assert C.PRESETS[C.DEFAULT_PRESET]["batched"] is True


def test_sequential_preset_uses_the_options_only_it_can():
    """"noisy" is the only sequential preset; it must actually exploit that."""
    noisy = C.PRESETS["noisy"]
    assert noisy["batched"] is False
    assert noisy["temperature_fallback"] is True
    assert noisy["hallucination_silence_threshold"] is not None


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
