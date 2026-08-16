"""End-to-end test of the per-file loop with a stubbed engine.

This is the highest-value test in the suite: it is the one that would have caught
the defect where the GUI's "View result" button never enabled, because the old
engine emitted "Done: <name>" from a branch that was disabled by default.

No GPU, no CTranslate2, no model weights — the engine is injected.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from whisper_core import WhisperCore
from whisper_engine.config import TranscriptionConfig
from whisper_engine.progress import parse_status
from whisper_engine.types import TranscriptResult, TranscriptSegment


class StubEngine:
    """Mimics WhisperEngine's contract, including the stop-mid-generator path."""

    def __init__(self, on_log=None, segments_per_file=3, fail_on=()):
        self.on_log = on_log
        self.segments_per_file = segments_per_file
        self.fail_on = set(fail_on)
        self.calls: list[Path] = []
        self.closed = False

    def unload(self):
        pass

    def transcribe_file(self, path, config, on_segment=None, on_percent=None,
                        on_stage=None, should_stop=None):
        self.calls.append(path)
        if path.name in self.fail_on:
            raise RuntimeError("boom")

        stop = should_stop or (lambda: False)
        for stage in ("model", "decode", "vad", "lang"):
            if on_stage:
                on_stage(stage)

        collected, stopped = [], False
        for i in range(self.segments_per_file):
            if stop():
                stopped = True
                self.closed = True
                break
            seg = TranscriptSegment(index=i + 1, start=float(i), end=float(i + 1),
                                    text=f"Сегмент {i + 1}.")
            collected.append(seg)
            if on_segment:
                on_segment(seg)
            if on_percent:
                on_percent(int((i + 1) / self.segments_per_file * 100))

        return TranscriptResult(
            source_path=path, segments=tuple(collected), language="ru",
            duration=float(self.segments_per_file), duration_after_vad=1.0,
            model_id=config.model_id, engine_info={"engine": "stub"},
            options=config.to_options_dict(),
            created_at=datetime.now(UTC), stopped_early=stopped,
        )


@pytest.fixture
def harness(tmp_path):
    """Returns (core, events, engine, config, files)."""
    audio = tmp_path / "audio"
    audio.mkdir()
    files = []
    for name in ("a.m4a", "b.wav"):
        p = audio / name
        p.write_bytes(b"")
        files.append(p)

    events, segments, results = [], [], []
    engine = StubEngine()

    core = WhisperCore(
        on_log=lambda m: None,
        on_progress=lambda pct, status: events.append((pct, status)),
        on_segment=segments.append,
        on_result=lambda p, w: results.append((p, w)),
        engine_factory=lambda on_log=None: engine,
    )
    config = TranscriptionConfig(device="cpu", output_dir=tmp_path / "out",
                                 formats=("txt", "json"))
    return core, events, engine, config, files, segments, results


def statuses(events):
    return [s for _, s in events]


# --- the regression this suite exists for ---------------------------------
def test_done_status_emitted_for_every_file(harness):
    core, events, _, config, files, _, _ = harness
    core.process_files(files, config)
    done = [parse_status(s).name for s in statuses(events) if parse_status(s).kind == "done_file"]
    assert done == ["a.m4a", "b.wav"], "every completed file must report 'Done: <name>'"


def test_status_sequence_is_well_formed(harness):
    core, events, _, config, files, _, _ = harness
    core.process_files(files, config)
    kinds = [parse_status(s).kind for s in statuses(events)]
    assert kinds[0] == "processing"
    assert "stage" in kinds and "percent" in kinds
    assert kinds[-1] == "done_all"
    assert "other" not in kinds, "the GUI must be able to parse every emitted status"


def test_progress_fraction_is_monotonic(harness):
    core, events, _, config, files, _, _ = harness
    core.process_files(files, config)
    fractions = [p for p, _ in events]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


# --- results and writing ---------------------------------------------------
def test_on_result_reports_real_written_paths(harness):
    """The GUI opens what this reports, so it must not assume a .txt exists."""
    core, _, _, config, files, _, results = harness
    core.process_files(files, config)
    assert len(results) == 2
    for source, written in results:
        assert {p.suffix for p in written} == {".txt", ".json"}
        assert all(p.exists() for p in written)
        assert all(p.stem == source.stem for p in written)


def test_segments_stream_to_the_callback(harness):
    core, _, _, config, files, segments, _ = harness
    core.process_files(files, config)
    assert len(segments) == 6
    assert segments[0].text == "Сегмент 1."


def test_nothing_written_when_no_speech(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    out = tmp_path / "out"
    core = WhisperCore(engine_factory=lambda on_log=None: StubEngine(segments_per_file=0))
    core.process_files([audio], TranscriptionConfig(device="cpu", output_dir=out))
    assert list(out.iterdir()) == []


# --- stopping --------------------------------------------------------------
def test_stop_writes_a_partial_and_halts_the_batch(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    files = []
    for name in ("a.wav", "b.wav"):
        p = audio_dir / name
        p.write_bytes(b"")
        files.append(p)

    out = tmp_path / "out"
    engine = StubEngine(segments_per_file=10)
    core = WhisperCore(engine_factory=lambda on_log=None: engine)

    # Stop as soon as the first segment has been handed over.
    core.on_segment = lambda seg: core.request_stop()

    core.process_files(files, TranscriptionConfig(device="cpu", output_dir=out))

    assert engine.closed, "the segment generator must be closed on stop"
    assert len(engine.calls) == 1, "the batch must not continue to the next file"
    names = {p.name for p in out.iterdir()}
    assert names == {"a.partial.txt"}, "a partial transcript beats nothing"


def test_stop_before_start_processes_nothing(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"")
    engine = StubEngine()
    core = WhisperCore(engine_factory=lambda on_log=None: engine)
    core.process_files([], TranscriptionConfig(device="cpu", output_dir=tmp_path / "out"))
    assert engine.calls == []


# --- resilience ------------------------------------------------------------
def test_one_failing_file_does_not_end_the_batch(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    files = []
    for name in ("bad.wav", "good.wav"):
        p = audio_dir / name
        p.write_bytes(b"")
        files.append(p)

    events = []
    engine = StubEngine(fail_on={"bad.wav"})
    core = WhisperCore(on_progress=lambda pct, s: events.append(s),
                       engine_factory=lambda on_log=None: engine)
    core.process_files(files, TranscriptionConfig(device="cpu", output_dir=tmp_path / "out"))

    done = [parse_status(s).name for s in events if parse_status(s).kind == "done_file"]
    assert done == ["good.wav"]
    assert len(engine.calls) == 2


def test_output_directory_is_created_before_the_model_loads(tmp_path):
    """A slow or failing load must not cost us the ability to save."""
    out = tmp_path / "deep" / "out"

    class ExplodingEngine(StubEngine):
        def transcribe_file(self, *a, **kw):
            assert out.is_dir(), "output dir must exist before the engine is touched"
            return super().transcribe_file(*a, **kw)

    p = tmp_path / "a.wav"
    p.write_bytes(b"")
    core = WhisperCore(engine_factory=lambda on_log=None: ExplodingEngine())
    core.process_files([p], TranscriptionConfig(device="cpu", output_dir=out))
    assert out.is_dir()


def test_config_notes_are_logged(tmp_path):
    """Silently-dropped options must surface, not just be silently dropped."""
    logs = []
    p = tmp_path / "a.wav"
    p.write_bytes(b"")
    core = WhisperCore(on_log=logs.append, engine_factory=lambda on_log=None: StubEngine())
    config = TranscriptionConfig(device="cpu", output_dir=tmp_path / "out",
                                 batched=True, condition_on_previous_text=True)
    core.process_files([p], config)
    assert any("batched mode ignores" in line for line in logs)
