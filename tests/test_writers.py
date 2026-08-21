import json

import pytest

from whisper_engine.writers import (
    WRITERS,
    pick_preview,
    write_all,
    write_json,
    write_markdown,
    write_srt,
    write_txt,
    write_vtt,
)


# --- TXT -------------------------------------------------------------------
def test_txt_splits_paragraph_on_long_gap(sample_result, tmp_path):
    out = write_txt(sample_result, tmp_path / "a.txt")
    body = out.read_text(encoding="utf-8")
    assert body.count("\n\n") == 1, "the >2 s gap should start exactly one new paragraph"
    first, second = body.strip().split("\n\n")
    assert first == "Первое предложение. Второе предложение."
    assert second == "После долгой паузы."


def test_txt_content_survives_whitespace_normalization(sample_result, tmp_path):
    """Paragraphing must not change the transcript for WER purposes."""
    import re
    body = write_txt(sample_result, tmp_path / "a.txt").read_text(encoding="utf-8")
    assert re.sub(r"\s+", " ", body).strip() == sample_result.text


def test_txt_on_empty_result(empty_result, tmp_path):
    assert write_txt(empty_result, tmp_path / "a.txt").read_text(encoding="utf-8") == ""


# --- SRT / VTT -------------------------------------------------------------
def test_srt_structure(sample_result, tmp_path):
    lines = write_srt(sample_result, tmp_path / "a.srt").read_text(encoding="utf-8").split("\n")
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,500"
    assert lines[2] == "Первое предложение."
    assert lines[3] == ""
    assert lines[4] == "2", "cue counter must be 1-based and contiguous"


def test_srt_cue_count_matches_nonempty_segments(sample_result, tmp_path):
    body = write_srt(sample_result, tmp_path / "a.srt").read_text(encoding="utf-8")
    assert len([ln for ln in body.split("\n") if "-->" in ln]) == 3


def test_zero_length_cue_is_stretched(make_segment, sample_result, tmp_path):
    """A player silently drops a cue whose end equals its start."""
    from dataclasses import replace
    result = replace(sample_result, segments=(make_segment(1, 4.0, 4.0, "Миг."),))
    body = write_srt(result, tmp_path / "a.srt").read_text(encoding="utf-8")
    assert "00:00:04,000 --> 00:00:04,100" in body


def test_vtt_header_and_separator(sample_result, tmp_path):
    body = write_vtt(sample_result, tmp_path / "a.vtt").read_text(encoding="utf-8")
    assert body.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:02.500" in body
    assert "," not in body.split("\n")[2]


def test_vtt_escapes_markup(make_segment, sample_result, tmp_path):
    from dataclasses import replace
    result = replace(sample_result, segments=(make_segment(1, 0.0, 1.0, "a < b & c > d"),))
    body = write_vtt(result, tmp_path / "a.vtt").read_text(encoding="utf-8")
    assert "a &lt; b &amp; c &gt; d" in body


# --- JSON ------------------------------------------------------------------
def test_json_round_trips_and_has_required_keys(sample_result, tmp_path):
    data = json.loads(write_json(sample_result, tmp_path / "a.json").read_text(encoding="utf-8"))
    for key in ("schema", "schema_version", "source", "model", "language",
                "options", "created_at", "stopped_early", "segments", "text"):
        assert key in data
    assert data["source"]["duration_sec"] == 900.0
    assert data["model"]["id"] == "whisper-medium"
    assert data["language"]["code"] == "ru"
    assert len(data["segments"]) == 3


def test_json_omits_words_key_when_absent(sample_result, tmp_path):
    """Consumers should be able to use `"words" in segment` rather than a null check."""
    data = json.loads(write_json(sample_result, tmp_path / "a.json").read_text(encoding="utf-8"))
    assert "words" not in data["segments"][0]
    assert "words" in data["segments"][2]
    assert data["segments"][2]["words"][0]["word"] == "После"


def test_json_is_not_ascii_escaped(sample_result, tmp_path):
    raw = write_json(sample_result, tmp_path / "a.json").read_text(encoding="utf-8")
    assert "Первое" in raw and "\\u041f" not in raw


def test_json_rounds_floats(make_segment, sample_result, tmp_path):
    from dataclasses import replace
    result = replace(sample_result,
                     segments=(make_segment(1, 0.123456789, 1.987654321, "x"),))
    data = json.loads(write_json(result, tmp_path / "a.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["start"] == 0.123
    assert data["segments"][0]["end"] == 1.988


# --- Markdown --------------------------------------------------------------
def test_markdown_has_metadata_table_and_transcript(sample_result, tmp_path):
    body = write_markdown(sample_result, tmp_path / "a.md").read_text(encoding="utf-8")
    assert body.startswith("# meeting")
    assert "| Source | `meeting.m4a` |" in body
    assert "| Duration | 0:15:00 |" in body
    assert "| Model | `whisper-medium` |" in body
    assert "## Transcript" in body
    assert "**[0:00:00]** Первое предложение. Второе предложение." in body


def test_markdown_marks_partial_runs(sample_result, tmp_path):
    from dataclasses import replace
    body = write_markdown(replace(sample_result, stopped_early=True),
                          tmp_path / "a.md").read_text(encoding="utf-8")
    assert "stopped early" in body


def test_markdown_survives_empty_segments(empty_result, tmp_path):
    body = write_markdown(empty_result, tmp_path / "a.md").read_text(encoding="utf-8")
    assert "## Transcript" in body


def test_markdown_buckets_by_five_minutes(make_segment, sample_result, tmp_path):
    from dataclasses import replace
    result = replace(sample_result, segments=(
        make_segment(1, 10.0, 12.0, "Начало."),
        make_segment(2, 310.0, 312.0, "Через пять минут."),
    ))
    body = write_markdown(result, tmp_path / "a.md").read_text(encoding="utf-8")
    assert "### 0:00:00" in body
    assert "### 0:05:00" in body


# --- Registry --------------------------------------------------------------
def test_write_all_produces_exactly_the_requested_formats(sample_result, tmp_path):
    written = write_all(sample_result, tmp_path, ["txt", "json"])
    assert {p.suffix for p in written} == {".txt", ".json"}
    assert {p.name for p in tmp_path.iterdir()} == {"meeting.txt", "meeting.json"}


def test_write_all_ignores_unknown_formats(sample_result, tmp_path):
    """A stale settings file must not abort a transcription that already succeeded."""
    written = write_all(sample_result, tmp_path, ["txt", "docx"])
    assert [p.suffix for p in written] == [".txt"]


def test_write_all_deduplicates_and_creates_dir(sample_result, tmp_path):
    target = tmp_path / "nested" / "out"
    written = write_all(sample_result, target, ["srt"])
    assert written[0].exists() and target.is_dir()


@pytest.mark.parametrize("fmt", list(WRITERS))
def test_every_registered_writer_runs(sample_result, tmp_path, fmt):
    assert write_all(sample_result, tmp_path, [fmt])[0].exists()


def test_pick_preview_prefers_readable_formats(tmp_path):
    paths = [tmp_path / "a.json", tmp_path / "a.srt", tmp_path / "a.txt"]
    assert pick_preview(paths).suffix == ".txt"
    assert pick_preview(paths[:2]).suffix == ".srt"
    assert pick_preview([]) is None
