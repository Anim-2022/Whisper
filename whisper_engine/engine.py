"""faster-whisper adapter.

This is the only module that imports faster_whisper, and it registers the CUDA
DLL directories before doing so — see :mod:`whisper_engine.cuda_dlls` for why
both mechanisms there are required.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import TranscriptionConfig
from .cuda_dlls import describe as describe_cuda
from .cuda_dlls import ensure_cuda_dlls
from .models import resolve_ct2_model_dir
from .types import TranscriptResult, TranscriptSegment, from_fw_segment

# Must happen before faster_whisper pulls in ctranslate2.
ensure_cuda_dlls()

LogCallback = Callable[[str], None]
SegmentCallback = Callable[[TranscriptSegment], None]
StageCallback = Callable[[str], None]
ShouldStop = Callable[[], bool]

#: Minimum wall-clock gap between progress emissions. A three-hour file yields
#: thousands of segments; without throttling the GUI queue becomes the bottleneck.
PROGRESS_INTERVAL_SEC = 0.25


class WhisperEngine:
    """Loads a CTranslate2 Whisper model and transcribes files with it."""

    def __init__(self, on_log: LogCallback | None = None):
        self.on_log = on_log
        self.model = None
        self.runner = None            # model, or BatchedInferencePipeline wrapping it
        self.model_id: str | None = None
        self.device: str | None = None
        self.compute_type: str | None = None
        self.batched: bool | None = None
        # What the caller asked for, as opposed to what we ended up running on.
        # Without this, a CUDA request that fell back to CPU never matches the
        # loaded state again, so every remaining file in the batch would repeat
        # the slow failing CUDA load before falling back once more.
        self._requested_device: str | None = None
        self._requested_compute_type: str | None = None

    # -- helpers -----------------------------------------------------------
    def log(self, message: str) -> None:
        if self.on_log:
            self.on_log(message)

    @property
    def engine_version(self) -> str:
        try:
            import faster_whisper
            return getattr(faster_whisper, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            return "unknown"

    def engine_info(self) -> dict:
        return {
            "engine": "faster-whisper",
            "engine_version": self.engine_version,
            "device": self.device or "",
            "compute_type": self.compute_type or "",
        }

    # -- loading -----------------------------------------------------------
    def _matches(self, config: TranscriptionConfig) -> bool:
        """Is the loaded model already what this config asks for?

        Compares against what was *requested* last time, not what we ended up
        running on: after a CUDA failure the engine sits on CPU while the config
        still says cuda, and comparing the effective device would make every
        subsequent file retry the failing CUDA load.
        """
        return (self.runner is not None
                and self.model_id == config.model_id
                and self._requested_device == config.device
                and self._requested_compute_type == config.compute_type
                and self.batched == config.batched)

    def load(self, config: TranscriptionConfig, force: bool = False) -> None:
        """Load the model, falling back to CPU if CUDA cannot be brought up.

        A CUDA failure surfaces at construction or on the first encode; either way
        finishing the run slowly beats failing it, so the fallback is automatic
        and loud rather than fatal.
        """
        if self._matches(config) and not force:
            return

        from faster_whisper import BatchedInferencePipeline, WhisperModel

        model_dir = resolve_ct2_model_dir(config.model_id)
        self.log(f"Loading model: {config.model_id}")
        self.log(f"  path: {model_dir}")
        self.log(f"  {config.describe()}")

        device, compute_type = config.device, config.compute_type
        t0 = time.monotonic()
        try:
            model = WhisperModel(str(model_dir), device=device,
                                 compute_type=compute_type, local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            if device != "cuda":
                raise
            self.log(f"CUDA initialization failed: {exc}")
            self.log(describe_cuda())
            self.log("Falling back to CPU (int8). Install requirements-gpu.txt to use the GPU.")
            device, compute_type = "cpu", "int8"
            model = WhisperModel(str(model_dir), device=device,
                                 compute_type=compute_type, local_files_only=True)

        self.model = model
        self.runner = BatchedInferencePipeline(model=model) if config.batched else model
        self.model_id = config.model_id
        self.device = device
        self.compute_type = compute_type
        self.batched = config.batched
        self._requested_device = config.device
        self._requested_compute_type = config.compute_type
        self.log(f"Model ready on {device}/{compute_type} in {time.monotonic() - t0:.1f}s")

    def unload(self) -> None:
        self.runner = None
        self.model = None
        self.model_id = self.device = self.compute_type = None
        self._requested_device = self._requested_compute_type = None
        self.batched = None

    # -- transcription -----------------------------------------------------
    def _transcribe_kwargs(self, config: TranscriptionConfig) -> dict:
        from faster_whisper.vad import VadOptions

        kwargs: dict[str, Any] = {
            "language": config.language,
            "task": "transcribe",
            "beam_size": config.beam_size,
            "temperature": config.temperature,
            "compression_ratio_threshold": config.compression_ratio_threshold,
            "log_prob_threshold": config.log_prob_threshold,
            "no_speech_threshold": config.no_speech_threshold,
            "word_timestamps": config.word_timestamps,
            "vad_filter": config.vad_filter,
        }
        if config.initial_prompt:
            kwargs["initial_prompt"] = config.initial_prompt
        if config.max_new_tokens is not None:
            kwargs["max_new_tokens"] = config.max_new_tokens

        if config.vad_filter:
            kwargs["vad_parameters"] = VadOptions(
                threshold=config.vad_threshold,
                min_speech_duration_ms=config.vad_min_speech_ms,
                min_silence_duration_ms=config.vad_min_silence_ms,
                speech_pad_ms=config.vad_speech_pad_ms,
            )

        if config.batched:
            kwargs["batch_size"] = config.batch_size
        else:
            # Sequential-only options; __post_init__ has already cleared these
            # when batched is on, so this branch is the only place they apply.
            kwargs["condition_on_previous_text"] = config.condition_on_previous_text
            if config.hallucination_silence_threshold is not None:
                kwargs["hallucination_silence_threshold"] = config.hallucination_silence_threshold
        return kwargs

    def transcribe_file(
        self,
        path: Path,
        config: TranscriptionConfig,
        on_segment: SegmentCallback | None = None,
        on_percent: Callable[[int], None] | None = None,
        on_stage: StageCallback | None = None,
        should_stop: ShouldStop | None = None,
    ) -> TranscriptResult:
        """Transcribe one file. Returns whatever was decoded, even if stopped early."""
        from . import audio as audio_mod
        from .progress import STAGE_DECODE, STAGE_LANG, STAGE_MODEL

        stop = should_stop or (lambda: False)

        if on_stage:
            on_stage(STAGE_MODEL)
        self.load(config)
        if stop():
            return self._empty_result(path, config, stopped=True)

        # Decode explicitly rather than handing the path to transcribe(): it gives
        # a fast, clear failure on unreadable input and a place to boost quiet audio.
        if on_stage:
            on_stage(STAGE_DECODE)
        samples, gain = audio_mod.decode(path, boost_quiet=config.boost_quiet_audio)
        if gain != 1.0:
            self.log(f"  quiet recording detected, boosted by {gain:.1f}x")
        if stop():
            return self._empty_result(path, config, stopped=True)

        if on_stage:
            on_stage(STAGE_LANG)
        segments_iter, info = self.runner.transcribe(samples, **self._transcribe_kwargs(config))

        total = float(getattr(info, "duration", 0.0) or 0.0)
        after_vad = float(getattr(info, "duration_after_vad", total) or total)
        detected = str(getattr(info, "language", "") or "")
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        if config.auto_lang:
            self.log(f"  detected language: {detected} ({lang_prob:.0%})")

        collected: list[TranscriptSegment] = []
        stopped_early = False
        last_emit = 0.0

        for raw in segments_iter:
            if stop():
                stopped_early = True
                # Close the generator so faster-whisper releases its buffers now
                # rather than whenever the object is collected.
                close = getattr(segments_iter, "close", None)
                if callable(close):
                    close()
                break

            seg = from_fw_segment(raw, len(collected) + 1)
            collected.append(seg)
            if on_segment:
                on_segment(seg)

            now = time.monotonic()
            if on_percent and total > 0 and (now - last_emit) >= PROGRESS_INTERVAL_SEC:
                on_percent(int(min(1.0, seg.end / total) * 100))
                last_emit = now

        if on_percent and not stopped_early:
            on_percent(100)

        return TranscriptResult(
            source_path=path,
            segments=tuple(collected),
            language=detected or (config.lang if not config.auto_lang else ""),
            language_probability=lang_prob,
            auto_detected_language=config.auto_lang,
            duration=total,
            duration_after_vad=after_vad,
            model_id=config.model_id,
            engine_info=self.engine_info(),
            options=config.to_options_dict(),
            created_at=datetime.now(UTC).astimezone(),
            stopped_early=stopped_early,
        )

    def _empty_result(self, path: Path, config: TranscriptionConfig,
                      stopped: bool = False) -> TranscriptResult:
        return TranscriptResult(
            source_path=path,
            segments=(),
            language="" if config.auto_lang else config.lang,
            auto_detected_language=config.auto_lang,
            model_id=config.model_id,
            engine_info=self.engine_info(),
            options=config.to_options_dict(),
            created_at=datetime.now(UTC).astimezone(),
            stopped_early=stopped,
        )


def default_engine_factory(on_log: LogCallback | None = None) -> WhisperEngine:
    """Injection point: tests pass a stub factory to run the core without a GPU."""
    return WhisperEngine(on_log=on_log)
