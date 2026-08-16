"""Timestamp formatting for subtitle and report writers.

Kept separate from :mod:`whisper_engine.writers` so it can be exercised without
constructing a transcript, and free of any faster-whisper import so it runs in CI.

faster_whisper ships its own ``format_timestamp``, but it omits the hours field
for media shorter than an hour, which is invalid SRT. This module always emits
``HH:MM:SS`` and carries the rounding fix from the original implementation.
"""
from __future__ import annotations


def format_timestamp(seconds: float, sep: str = ",", always_hours: bool = True) -> str:
    """Format ``seconds`` as ``HH:MM:SS<sep>mmm``.

    ``sep=","`` gives the SRT form, ``sep="."`` the WebVTT form.

    The carry branch is not theoretical: ``round((3.9996 - 3) * 1000)`` is 1000,
    which would otherwise render as the invalid ``00:00:03,1000``.
    """
    if seconds is None or seconds < 0:
        seconds = 0.0

    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms == 1000:
        ms = 0
        whole += 1

    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)

    if hours or always_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"
    return f"{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def format_clock(seconds: float) -> str:
    """Format ``seconds`` as ``H:MM:SS`` for headings and metadata tables."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    whole = int(round(seconds))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """Human-readable duration for logs, e.g. ``2h 52m`` or ``6m 00s``."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    whole = int(round(seconds))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
