"""Audio loading.

Decoding goes through faster-whisper's PyAV-based ``decode_audio``, which bundles
FFmpeg and therefore reads m4a, mp4, webm, mkv, opus, aac and wma without any
system dependency — the formats the previous soundfile-only path could not open
at all, and exactly what Zoom and Teams produce.

The previous engine also RMS-normalized every file to a target dBFS and rescaled
the peak. That is dropped: Whisper normalizes its mel spectrogram per window and
was trained on unnormalized audio, so the pass cost two traversals of a
multi-hundred-megabyte array for no accuracy benefit. Only a guard against
pathologically quiet recordings survives.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

#: Below this peak the recording is very quiet (~-34 dBFS) — a phone in a pocket,
#: or a badly gain-staged conference mic.
QUIET_PEAK_THRESHOLD = 0.02
#: Target peak when boosting. Deliberately modest: the goal is to lift the signal
#: out of the noise floor, not to maximize it.
QUIET_TARGET_PEAK = 0.1

#: Container and codec suffixes PyAV handles. Superset of the previous glob list,
#: which claimed .m4a support the decoder did not actually have.
AUDIO_SUFFIXES = frozenset({
    ".mp3", ".wav", ".flac", ".m4a", ".mp4", ".ogg", ".opus", ".aac", ".wma",
    ".webm", ".mkv", ".mov", ".avi", ".aiff", ".aif", ".wv", ".amr", ".3gp", ".m4b",
})


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES


def find_audio_files(directory: Path) -> list[Path]:
    """Audio files directly inside ``directory``, sorted by name.

    One pass over the directory rather than a glob per extension, so a file can
    never be enqueued twice by two overlapping patterns.
    """
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if is_audio_file(p))


def maybe_boost_quiet(audio: np.ndarray, enabled: bool = True) -> tuple[np.ndarray, float]:
    """Scale up pathologically quiet audio. Returns ``(audio, gain_applied)``."""
    if not enabled or audio.size == 0:
        return audio, 1.0
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0 or peak >= QUIET_PEAK_THRESHOLD:
        return audio, 1.0
    gain = QUIET_TARGET_PEAK / peak
    return audio * gain, gain


def decode(path: Path, boost_quiet: bool = True) -> tuple[np.ndarray, float]:
    """Decode ``path`` to 16 kHz mono float32. Returns ``(samples, gain_applied)``.

    Imported lazily so this module stays importable without faster-whisper.
    """
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    audio = np.asarray(audio, dtype=np.float32)
    return maybe_boost_quiet(audio, enabled=boost_quiet)


def duration_of(audio: np.ndarray) -> float:
    return len(audio) / float(SAMPLE_RATE) if audio.size else 0.0
