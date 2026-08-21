"""Transcription configuration.

Replaces the previous dataclass, whose fields described a hand-rolled
VAD/chunk/batch/stitch loop that no longer exists. Fields here map onto
faster-whisper's own arguments, so there is no translation layer to drift.

Free of any faster-whisper import so it can be validated in CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .device import COMPUTE_TYPES, DEVICES, resolve_compute_type, resolve_device

DEFAULT_MODEL = "whisper-medium"

#: Temperature ladder faster-whisper walks when a decode looks degenerate.
TEMPERATURE_LADDER: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class TranscriptionConfig:
    """Everything one transcription run needs.

    Defaults are the measured-best configuration for long Russian meeting audio
    on a 16 GB consumer GPU: whisper-medium in batched mode reaches ~60-70x
    realtime while matching or beating the previous engine on punctuation and on
    retention of English technical terms. large-v3-turbo is faster still but is
    distilled down to four decoder layers and loses punctuation badly, so it is
    an option rather than the default.
    """

    # -- model / compute ---------------------------------------------------
    model_id: str = DEFAULT_MODEL
    device: str = "auto"
    compute_type: str = "auto"

    # -- language ----------------------------------------------------------
    lang: str = "ru"
    auto_lang: bool = False
    #: Seeds the decoder with vocabulary and punctuation style. Attendee names and
    #: product jargon here measurably improve their spelling in the output.
    initial_prompt: str | None = None

    # -- decoding ----------------------------------------------------------
    #: Batched packs VAD chunks into parallel 30 s windows: several times faster,
    #: but faster-whisper disables sequential-only options in this mode (see
    #: __post_init__).
    batched: bool = True
    batch_size: int = 16
    beam_size: int = 5
    temperature_fallback: bool = True
    #: Off by default despite faster-whisper's own default: carrying previous text
    #: into the next window is the classic cause of repetition loops on long
    #: meeting audio, which is exactly this application's workload.
    condition_on_previous_text: bool = False
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    hallucination_silence_threshold: float | None = None
    word_timestamps: bool = False
    max_new_tokens: int | None = None

    # -- VAD ---------------------------------------------------------------
    vad_filter: bool = True
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    #: faster-whisper's own default is 2000 ms, which merges across natural
    #: sentence breaks; presets set this explicitly.
    vad_min_silence_ms: int = 700
    vad_speech_pad_ms: int = 400

    # -- audio -------------------------------------------------------------
    #: Scale up audio peaking below ~-34 dBFS (a phone in a pocket). Whisper is
    #: trained on unnormalized audio, so this only rescues pathological input.
    boost_quiet_audio: bool = True

    # -- output ------------------------------------------------------------
    formats: tuple[str, ...] = ("txt",)
    output_dir: Path = Path("./audio_to_text")

    #: Warnings raised while normalizing the config, for the caller to log.
    notes: tuple[str, ...] = field(default_factory=tuple, compare=False)

    def __post_init__(self):
        # Validate what was *requested* before resolution, because resolve_*
        # coerces to something valid and would otherwise mask a typo — silently
        # running settings other than the ones asked for.
        requested_device = (self.device or "auto").strip().lower()
        if requested_device not in DEVICES and requested_device != "auto":
            raise ValueError(f"device must be one of {DEVICES} or 'auto', got {self.device!r}")
        requested_compute = (self.compute_type or "auto").strip().lower()
        if requested_compute not in COMPUTE_TYPES:
            raise ValueError(
                f"compute_type must be one of {COMPUTE_TYPES}, got {self.compute_type!r}")

        self.device = resolve_device(self.device)
        self.compute_type = resolve_compute_type(self.compute_type, self.device)

        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.formats, str):
            self.formats = (self.formats,)
        self.formats = tuple(dict.fromkeys(f.lower().lstrip(".") for f in self.formats if f))
        if not self.formats:
            self.formats = ("txt",)

        if self.beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 < self.vad_threshold < 1.0:
            raise ValueError("vad_threshold must be between 0 and 1 (exclusive)")
        if self.vad_min_silence_ms < 0 or self.vad_min_speech_ms < 0 or self.vad_speech_pad_ms < 0:
            raise ValueError("VAD durations must be non-negative")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1 or None")

        notes = list(self.notes)

        # BatchedInferencePipeline derives its clip timestamps from VAD and
        # raises "No clip timestamps found" without it. Turning VAD back on would
        # contradict an explicit choice, so honour the choice and drop to the
        # sequential path, which transcribes the audio unsegmented as asked.
        if self.batched and not self.vad_filter:
            self.batched = False
            notes.append("batched mode requires VAD; switching to sequential mode")

        # faster-whisper applies the hallucination filter only inside the
        # word-timestamp code path, so asking for one without the other silently
        # does nothing. Couple them rather than let the UI claim it is active.
        if self.hallucination_silence_threshold is not None and not self.word_timestamps:
            self.word_timestamps = True
            notes.append("hallucination filter requires word timestamps; enabling them")

        # BatchedInferencePipeline discards these regardless of what we pass.
        # Null them here so logs and JSON metadata describe the real run.
        if self.batched:
            dropped = []
            if self.condition_on_previous_text:
                self.condition_on_previous_text = False
                dropped.append("condition_on_previous_text")
            if self.hallucination_silence_threshold is not None:
                self.hallucination_silence_threshold = None
                dropped.append("hallucination_silence_threshold")
            if self.temperature_fallback:
                self.temperature_fallback = False
                dropped.append("temperature_fallback")
            if dropped:
                notes.append("batched mode ignores " + ", ".join(dropped))

            # Batched decoding emits one segment per packed 30 s window, so
            # subtitle cues come out ~30 s long — fine as a transcript, useless
            # as actual subtitles. Sequential yields sentence-level segments.
            if {"srt", "vtt"} & set(self.formats):
                notes.append("batched mode produces ~30 s subtitle cues; "
                             "use the sequential mode for sentence-level timing")

        self.notes = tuple(notes)

    # -- derived -----------------------------------------------------------
    @property
    def temperature(self):
        """Value for faster-whisper's ``temperature`` argument."""
        return TEMPERATURE_LADDER if self.temperature_fallback else 0.0

    @property
    def language(self) -> str | None:
        """``None`` asks faster-whisper to detect the language."""
        return None if self.auto_lang else self.lang

    def describe(self) -> str:
        mode = f"batched(bs={self.batch_size})" if self.batched else "sequential"
        lang = "auto" if self.auto_lang else self.lang
        return (f"{self.model_id} on {self.device}/{self.compute_type}, {mode}, "
                f"beam={self.beam_size}, lang={lang}, vad={'on' if self.vad_filter else 'off'}")

    def to_options_dict(self) -> dict:
        """Flat snapshot of the decode-affecting options, for JSON metadata."""
        return {
            "batched": self.batched,
            "batch_size": self.batch_size if self.batched else None,
            "beam_size": self.beam_size,
            "temperature_fallback": self.temperature_fallback,
            "condition_on_previous_text": self.condition_on_previous_text,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "word_timestamps": self.word_timestamps,
            "initial_prompt": self.initial_prompt,
            "vad_filter": self.vad_filter,
            "vad_threshold": self.vad_threshold,
            "vad_min_speech_ms": self.vad_min_speech_ms,
            "vad_min_silence_ms": self.vad_min_silence_ms,
            "vad_speech_pad_ms": self.vad_speech_pad_ms,
            "boost_quiet_audio": self.boost_quiet_audio,
        }
