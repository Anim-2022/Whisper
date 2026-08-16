"""Output writers: TXT, SRT, VTT, JSON, Markdown.

All five share the signature ``(result, out_path) -> Path`` and are registered in
:data:`WRITERS`, so adding a format means adding one function and one entry.
Nothing here imports faster-whisper, which keeps the whole module unit-testable.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from .timestamps import format_clock, format_timestamp
from .types import TranscriptResult, TranscriptSegment

# A pause longer than this starts a new paragraph in TXT and may start a new
# bucket heading in Markdown. Chosen to match a speaker's breath/topic break
# rather than the gaps VAD leaves between sentences.
PARAGRAPH_GAP_SEC = 2.0

# Markdown groups the transcript under headings this many seconds apart.
MARKDOWN_BUCKET_SEC = 300.0

# Subtitle cues shorter than this are stretched so players do not drop them.
MIN_CUE_SEC = 0.1

JSON_SCHEMA_NAME = "whisper-unified/transcript"
JSON_SCHEMA_VERSION = 1


def _nonempty(segments: Sequence[TranscriptSegment]) -> list[TranscriptSegment]:
    return [s for s in segments if s.text.strip()]


def _cue_bounds(seg: TranscriptSegment) -> tuple[float, float]:
    start = max(0.0, seg.start)
    end = max(seg.end, start + MIN_CUE_SEC)
    return start, end


def _paragraphs(segments: Sequence[TranscriptSegment]) -> Iterable[list[TranscriptSegment]]:
    """Group segments into paragraphs, breaking on pauses."""
    current: list[TranscriptSegment] = []
    prev_end = None
    for seg in segments:
        if prev_end is not None and (seg.start - prev_end) > PARAGRAPH_GAP_SEC and current:
            yield current
            current = []
        current.append(seg)
        prev_end = seg.end
    if current:
        yield current


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_txt(result: TranscriptResult, out_path: Path) -> Path:
    """Plain text, split into paragraphs on long pauses.

    Paragraphing does not affect WER comparisons: evaluate_transcriptions.py
    collapses all whitespace before scoring.
    """
    segments = _nonempty(result.segments)
    blocks = [" ".join(s.text.strip() for s in para) for para in _paragraphs(segments)]
    body = "\n\n".join(b for b in blocks if b)
    out_path.write_text(body + ("\n" if body else ""), encoding="utf-8")
    return out_path


def write_srt(result: TranscriptResult, out_path: Path) -> Path:
    """SubRip. 1-based counter, HH:MM:SS,mmm, blank line between cues."""
    lines: list[str] = []
    for i, seg in enumerate(_nonempty(result.segments), 1):
        start, end = _cue_bounds(seg)
        lines.append(str(i))
        lines.append(f"{format_timestamp(start, ',')} --> {format_timestamp(end, ',')}")
        lines.append(seg.text.strip())
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _vtt_escape(text: str) -> str:
    # Only these three matter for WebVTT cue payloads.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_vtt(result: TranscriptResult, out_path: Path) -> Path:
    """WebVTT. Same cues as SRT but with a header and a dot separator."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in _nonempty(result.segments):
        start, end = _cue_bounds(seg)
        lines.append(f"{format_timestamp(start, '.')} --> {format_timestamp(end, '.')}")
        lines.append(_vtt_escape(seg.text.strip()))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_json(result: TranscriptResult, out_path: Path) -> Path:
    """Machine-readable transcript with timings and per-segment quality metrics."""
    segments = []
    for seg in _nonempty(result.segments):
        entry = {
            "id": seg.index,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "no_speech_prob": round(seg.no_speech_prob, 4),
            "avg_logprob": round(seg.avg_logprob, 4),
            "compression_ratio": round(seg.compression_ratio, 4),
            "temperature": round(seg.temperature, 3),
        }
        # Omit the key entirely rather than emitting null, so consumers can use
        # a plain `"words" in segment` test.
        if seg.words:
            entry["words"] = [
                {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.word,
                    "probability": round(w.probability, 4),
                }
                for w in seg.words
            ]
        segments.append(entry)

    payload = {
        "schema": JSON_SCHEMA_NAME,
        "schema_version": JSON_SCHEMA_VERSION,
        "source": {
            "file": result.source_path.name,
            "duration_sec": round(result.duration, 3),
            "duration_after_vad_sec": round(result.duration_after_vad, 3),
        },
        "model": dict(result.engine_info, id=result.model_id),
        "language": {
            "code": result.language,
            "probability": round(result.language_probability, 4),
            "auto_detected": result.auto_detected_language,
        },
        "options": result.options,
        "created_at": result.created_at.isoformat() if result.created_at else None,
        "stopped_early": result.stopped_early,
        "segments": segments,
        "text": result.text,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_markdown(result: TranscriptResult, out_path: Path) -> Path:
    """Readable meeting protocol: metadata table, then time-bucketed transcript.

    A future summarizer would insert its section between the table and the
    transcript; nothing else in the pipeline would need to change.
    """
    lines: list[str] = [f"# {result.source_path.stem}", ""]

    lang = result.language or "?"
    if result.auto_detected_language and result.language_probability:
        lang = f"{lang} (auto, {result.language_probability:.0%})"

    lines += [
        "| | |",
        "|---|---|",
        f"| Source | `{result.source_path.name}` |",
        f"| Duration | {format_clock(result.duration)} |",
        f"| Speech | {format_clock(result.duration_after_vad)} |",
        f"| Language | {lang} |",
        f"| Model | `{result.model_id}` |",
    ]
    if result.created_at:
        lines.append(f"| Transcribed | {result.created_at:%Y-%m-%d %H:%M} |")
    if result.stopped_early:
        lines.append("| Status | **stopped early — partial transcript** |")
    lines += ["", "## Transcript", ""]

    segments = _nonempty(result.segments)
    current_bucket = -1
    for para in _paragraphs(segments):
        bucket = int(para[0].start // MARKDOWN_BUCKET_SEC)
        if bucket != current_bucket:
            current_bucket = bucket
            lines += ["", f"### {format_clock(bucket * MARKDOWN_BUCKET_SEC)}", ""]
        text = " ".join(s.text.strip() for s in para)
        lines.append(f"**[{format_clock(para[0].start)}]** {text}")
        lines.append("")

    lines += ["---", "", "*Generated offline by Whisper Unified.*", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
WRITERS: dict[str, tuple[str, Callable[[TranscriptResult, Path], Path]]] = {
    "txt": (".txt", write_txt),
    "srt": (".srt", write_srt),
    "vtt": (".vtt", write_vtt),
    "json": (".json", write_json),
    "md": (".md", write_markdown),
}

#: Order used when the GUI picks something to show the user after a run.
PREVIEW_PREFERENCE: tuple[str, ...] = (".txt", ".md", ".srt", ".vtt", ".json")


def available_formats() -> list[str]:
    return list(WRITERS)


def write_all(result: TranscriptResult, out_dir: Path,
              formats: Sequence[str], stem: str = "") -> list[Path]:
    """Write ``result`` in every requested format. Returns the paths written.

    Unknown format names are ignored rather than fatal: a stale settings file
    should not abort a transcription that already succeeded.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or result.source_path.stem
    written: list[Path] = []
    for fmt in formats:
        entry = WRITERS.get(str(fmt).lower().lstrip("."))
        if entry is None:
            continue
        suffix, writer = entry
        written.append(writer(result, out_dir / f"{stem}{suffix}"))
    return written


def pick_preview(paths: Sequence[Path]) -> Path | None:
    """Choose the most human-readable of the written files."""
    for suffix in PREVIEW_PREFERENCE:
        for p in paths:
            if p.suffix.lower() == suffix:
                return p
    return paths[0] if paths else None
