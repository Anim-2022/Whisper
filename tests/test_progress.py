"""The status wire format is a contract between the engine and the GUI.

Changing any string here without updating gui/status.py silently breaks per-file
tracking — which is exactly how the "View result" button came to be permanently
disabled in the previous engine.
"""
from __future__ import annotations

import pytest

from whisper_engine.progress import ProgressReporter, parse_status


@pytest.fixture
def recorder():
    events = []
    reporter = ProgressReporter(on_progress=lambda pct, status: events.append((pct, status)))
    reporter.start(4)
    return reporter, events


def test_emitted_strings_are_exact(recorder):
    reporter, events = recorder
    reporter.file_started(0, "a.m4a")
    reporter.stage(0, "decode", "a.m4a")
    reporter.file_percent(0, 50)
    reporter.file_done(0, "a.m4a")
    reporter.all_done()
    reporter.stopped()
    assert [s for _, s in events] == [
        "Processing a.m4a...", "stage:decode:a.m4a", "transcribing:50",
        "Done: a.m4a", "Done.", "Stopped",
    ]


def test_overall_fraction_accounts_for_file_index(recorder):
    reporter, events = recorder
    reporter.file_percent(2, 50)          # third of four files, halfway
    assert events[-1][0] == pytest.approx(0.625)


def test_fraction_is_clamped(recorder):
    reporter, events = recorder
    reporter.file_percent(0, 500)
    assert 0.0 <= events[-1][0] <= 1.0


def test_no_callback_is_safe():
    ProgressReporter().file_done(0, "a.wav")   # must not raise


def test_zero_total_does_not_divide_by_zero():
    events = []
    r = ProgressReporter(lambda p, s: events.append((p, s)))
    r.start(0)
    r.file_percent(0, 50)
    assert events[-1][0] == 0.0


# --- parse_status round-trip ----------------------------------------------
def test_parse_processing():
    p = parse_status("Processing meeting.m4a...")
    assert (p.kind, p.name) == ("processing", "meeting.m4a")


def test_parse_stage_keeps_filename():
    p = parse_status("stage:vad:meeting.m4a")
    assert (p.kind, p.stage, p.name) == ("stage", "vad", "meeting.m4a")


def test_parse_percent():
    assert parse_status("transcribing:73").percent == 73


def test_parse_percent_is_clamped_and_survives_garbage():
    assert parse_status("transcribing:900").percent == 100
    assert parse_status("transcribing:abc").percent == 0


def test_parse_done_file_is_distinct_from_done_all():
    assert parse_status("Done: meeting.m4a").kind == "done_file"
    assert parse_status("Done: meeting.m4a").name == "meeting.m4a"
    assert parse_status("Done.").kind == "done_all"


def test_parse_stopped_and_unknown():
    assert parse_status("Stopped").kind == "stopped"
    assert parse_status("something else").kind == "other"
    assert parse_status("").kind == "other"


def test_every_reporter_string_parses_back(recorder):
    """Round-trip guard: the reporter must not emit anything the GUI cannot read."""
    reporter, events = recorder
    reporter.file_started(0, "x.wav")
    reporter.stage(0, "model", "x.wav")
    reporter.file_percent(0, 10)
    reporter.file_done(0, "x.wav")
    reporter.all_done()
    reporter.stopped()
    assert all(parse_status(s).kind != "other" for _, s in events)


def test_filename_with_colons_survives_done_status():
    """Windows forbids ':' in names, but network paths and odd exports slip through."""
    assert parse_status("Done: a:b.wav").name == "a:b.wav"
