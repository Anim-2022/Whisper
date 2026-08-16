"""Transport types shared by the engine and the writers.

Deliberately free of any faster-whisper import: writers and their tests must run
on a machine with no CTranslate2 and no model weights. :func:`from_fw_segment` is
the single adapter that knows what a faster-whisper segment looks like, and it
only touches attributes, so a plain stub object satisfies it in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WordTS:
    """A single word with its timing and the model's confidence in it."""
    start: float
    end: float
    word: str
    probability: float = 0.0


@dataclass(frozen=True)
class TranscriptSegment:
    """One decoded span of speech.

    The quality fields are carried through to JSON so a bad run can be diagnosed
    after the fact: a high ``no_speech_prob`` together with confident-looking text
    is the signature of a hallucination over silence, and a ``compression_ratio``
    above ~2.4 marks a repetition loop.
    """
    index: int
    start: float
    end: float
    text: str
    words: tuple[WordTS, ...] = ()
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    temperature: float = 0.0


@dataclass(frozen=True)
class TranscriptResult:
    """Everything one audio file produced, ready to be written in any format."""
    source_path: Path
    segments: tuple[TranscriptSegment, ...]
    language: str = ""
    language_probability: float = 0.0
    auto_detected_language: bool = False
    duration: float = 0.0
    duration_after_vad: float = 0.0
    model_id: str = ""
    engine_info: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    stopped_early: bool = False

    @property
    def text(self) -> str:
        """Full transcript as one string, segments joined with single spaces."""
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def has_words(self) -> bool:
        return any(s.words for s in self.segments)


def from_fw_segment(seg: Any, index: int) -> TranscriptSegment:
    """Adapt one faster-whisper ``Segment`` into our own type.

    Attribute access only, with defaults for everything optional, so this keeps
    working across faster-whisper versions and accepts test stubs.
    """
    raw_words = getattr(seg, "words", None) or ()
    words = tuple(
        WordTS(
            start=float(getattr(w, "start", 0.0) or 0.0),
            end=float(getattr(w, "end", 0.0) or 0.0),
            word=str(getattr(w, "word", "") or ""),
            probability=float(getattr(w, "probability", 0.0) or 0.0),
        )
        for w in raw_words
    )
    return TranscriptSegment(
        index=index,
        start=float(getattr(seg, "start", 0.0) or 0.0),
        end=float(getattr(seg, "end", 0.0) or 0.0),
        text=str(getattr(seg, "text", "") or ""),
        words=words,
        no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
        avg_logprob=float(getattr(seg, "avg_logprob", 0.0) or 0.0),
        compression_ratio=float(getattr(seg, "compression_ratio", 0.0) or 0.0),
        temperature=float(getattr(seg, "temperature", 0.0) or 0.0),
    )
