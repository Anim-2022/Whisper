"""Shared fixtures.

Nothing here imports customtkinter or faster-whisper: the whole default suite
must run on a headless CI runner with neither Tk nor CUDA installed.
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whisper_engine.types import TranscriptResult, TranscriptSegment, WordTS  # noqa: E402


@pytest.fixture
def make_segment():
    def _make(index, start, end, text, words=(), **kw):
        return TranscriptSegment(
            index=index, start=start, end=end, text=text,
            words=tuple(words),
            no_speech_prob=kw.get("no_speech_prob", 0.01),
            avg_logprob=kw.get("avg_logprob", -0.25),
            compression_ratio=kw.get("compression_ratio", 1.4),
            temperature=kw.get("temperature", 0.0),
        )
    return _make


@pytest.fixture
def sample_result(make_segment):
    """A small transcript with a >2 s gap between segment 2 and 3.

    The gap matters: it is what the TXT paragraph split and the Markdown bucket
    logic key off, so tests can assert on it.
    """
    segments = (
        make_segment(1, 0.0, 2.5, "Первое предложение."),
        make_segment(2, 2.5, 5.0, "Второе предложение."),
        make_segment(3, 12.0, 15.0, "После долгой паузы.",
                     words=[WordTS(12.0, 12.6, "После", 0.98),
                            WordTS(12.6, 13.4, "долгой", 0.95),
                            WordTS(13.4, 15.0, "паузы.", 0.91)]),
    )
    return TranscriptResult(
        source_path=Path("meeting.m4a"),
        segments=segments,
        language="ru",
        language_probability=0.99,
        auto_detected_language=False,
        duration=900.0,
        duration_after_vad=700.0,
        model_id="whisper-medium",
        engine_info={"engine": "faster-whisper", "engine_version": "1.2.1",
                     "device": "cuda", "compute_type": "float16"},
        options={"beam_size": 5, "vad_filter": True},
        created_at=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
        stopped_early=False,
    )


@pytest.fixture
def empty_result():
    return TranscriptResult(source_path=Path("silence.wav"), segments=())


@pytest.fixture
def fake_ct2_dir(tmp_path):
    """Build a directory tree that looks like a converted CT2 model."""
    def _make(name="whisper-medium", root=None, omit=()):
        base = (root or tmp_path / "ct2")
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        for f in ("model.bin", "config.json", "tokenizer.json",
                  "preprocessor_config.json", "vocabulary.json"):
            if f in omit:
                continue
            (d / f).write_text("{}", encoding="utf-8")
        return d
    return _make
