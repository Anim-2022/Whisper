"""Batch orchestration over :mod:`whisper_engine`.

Thin facade: it owns the per-file loop, progress reporting and file writing, and
delegates everything acoustic to :class:`whisper_engine.engine.WhisperEngine`.

The previous version of this module was ~1000 lines implementing VAD, chunking,
batching, text stitching and recursive OOM splitting by hand. faster-whisper does
all of that natively and better, so that machinery is gone.

``TranscriptionConfig`` is re-exported here so existing importers keep working.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from whisper_engine.config import TranscriptionConfig
from whisper_engine.engine import WhisperEngine, default_engine_factory
from whisper_engine.progress import ProgressReporter
from whisper_engine.types import TranscriptResult, TranscriptSegment
from whisper_engine.writers import write_all

__all__ = ["TranscriptionConfig", "WhisperCore"]

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, str], None]
SegmentCallback = Callable[[TranscriptSegment], None]
ResultCallback = Callable[[Path, list[Path]], None]


class WhisperCore:
    """Transcribes a list of files, reporting progress as it goes."""

    def __init__(
        self,
        on_log: LogCallback | None = None,
        on_progress: ProgressCallback | None = None,
        on_segment: SegmentCallback | None = None,
        on_result: ResultCallback | None = None,
        engine_factory: Callable[..., WhisperEngine] = default_engine_factory,
    ):
        """
        Args:
            on_log: Receives log lines.
            on_progress: Receives (fraction 0..1, status string).
            on_segment: Receives each decoded segment as it arrives, for live display.
            on_result: Receives (audio_path, [written paths]) after each file.
                Reporting the real paths is what lets the GUI open the right file
                when the user has selected formats other than TXT.
            engine_factory: Injection point. Tests pass a stub so the whole loop
                runs without CUDA or model weights.
        """
        self.on_log = on_log
        self.on_segment = on_segment
        self.on_result = on_result
        self.reporter = ProgressReporter(on_progress)
        self.engine_factory = engine_factory

        self.stop_requested = False
        self.engine: WhisperEngine | None = None

    # -- plumbing ----------------------------------------------------------
    def log(self, message: str) -> None:
        if self.on_log:
            self.on_log(message)

    def request_stop(self) -> None:
        self.stop_requested = True

    def _should_stop(self) -> bool:
        return self.stop_requested

    def _ensure_engine(self) -> WhisperEngine:
        if self.engine is None:
            self.engine = self.engine_factory(on_log=self.on_log)
        return self.engine

    def unload(self) -> None:
        if self.engine is not None:
            self.engine.unload()

    # -- main loop ---------------------------------------------------------
    def process_files(self, file_paths: Sequence[Path], config: TranscriptionConfig) -> None:
        self.stop_requested = False
        files = list(file_paths)

        # Create the destination before touching the model: a slow or failing load
        # should not cost us the ability to save once we do have output.
        config.output_dir.mkdir(parents=True, exist_ok=True)

        for note in config.notes:
            self.log(f"Note: {note}")

        self.reporter.start(len(files))
        engine = self._ensure_engine()

        for index, path in enumerate(files):
            if self.stop_requested:
                break

            self.reporter.file_started(index, path.name)
            self.log(f"--- Started: {path.name} ---")

            try:
                result = engine.transcribe_file(
                    path,
                    config,
                    on_segment=self.on_segment,
                    on_percent=lambda pct, i=index: self.reporter.file_percent(i, pct),
                    on_stage=lambda stage, i=index, n=path.name: self.reporter.stage(i, stage, n),
                    should_stop=self._should_stop,
                )
            except Exception as exc:  # noqa: BLE001 - one bad file must not end the batch
                self.log(f"Error processing {path.name}: {type(exc).__name__}: {exc}")
                continue

            written = self._write_result(result, config)

            if result.stopped_early:
                self.log(f"Stopped during {path.name}; wrote partial transcript.")
                if self.on_result and written:
                    self.on_result(path, written)
                break

            # Order matters: the consumer reacts to the "Done" status by lighting
            # up its post-run actions, so it has to already know which files were
            # written. Reporting the paths first makes that ordering a property of
            # the core rather than of the consumer's queue-draining order.
            if self.on_result:
                self.on_result(path, written)

            # Unconditional on the success path. The absence of this emission on
            # the default path is what previously left the GUI's "View result"
            # button permanently disabled and its per-file ETA always empty.
            self.reporter.file_done(index, path.name)

        if self.stop_requested:
            self.log("Stopped by user.")
            self.reporter.stopped()
        else:
            self.reporter.all_done()
            self.log("All tasks completed.")

    # -- output ------------------------------------------------------------
    def _write_result(self, result: TranscriptResult, config: TranscriptionConfig) -> list[Path]:
        if not result.segments:
            self.log(f"No speech detected in {result.source_path.name}; nothing written.")
            return []

        # A partial run is marked in the filename so it is never mistaken for a
        # complete transcript later.
        stem = result.source_path.stem + (".partial" if result.stopped_early else "")
        written = write_all(result, config.output_dir, config.formats, stem=stem)
        for path in written:
            self.log(f"Saved: {path.name}")
        return written
