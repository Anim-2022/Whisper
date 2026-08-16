"""The single source of every progress status string.

Background: the GUI's per-file tracking keys off a ``"Done: <name>"`` status. In
the previous engine that string was emitted from exactly one branch — the HF
pipeline path, which was disabled by default — so on the default path the
"View result" button never enabled and the per-file ETA never accumulated.

The fix is structural rather than textual: only :class:`ProgressReporter` may
build these strings, and the per-file loop calls :meth:`file_done` on its success
path. A future branch cannot forget to emit it without deleting a call.

Wire formats are kept byte-identical to the previous engine so the GUI's existing
translator keeps working; ``stage:`` is the only addition.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: Stage keys emitted before the first segment arrives. On a multi-hour file the
#: decode/VAD/language-detect prelude can take a minute with nothing to show, and
#: without these the Stop button looks dead.
STAGE_MODEL = "model"
STAGE_DECODE = "decode"
STAGE_VAD = "vad"
STAGE_LANG = "lang"

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class ParsedStatus:
    """Structured view of a raw status string, for the GUI to render."""
    kind: str                      # "processing" | "stage" | "percent" | "done_file"
    #                                | "done_all" | "stopped" | "other"
    name: str = ""                 # file name, where applicable
    stage: str = ""                # one of the STAGE_* keys, for kind == "stage"
    percent: int = 0               # 0..100, for kind == "percent"
    raw: str = ""


def parse_status(status: str) -> ParsedStatus:
    """Inverse of :class:`ProgressReporter`. Pure, so it is testable without Tk."""
    s = (status or "").strip()

    if s.startswith("Processing ") and s.endswith("..."):
        return ParsedStatus("processing", name=s[len("Processing "):-3], raw=s)

    if s.startswith("stage:"):
        rest = s[len("stage:"):]
        stage, _, name = rest.partition(":")
        return ParsedStatus("stage", name=name, stage=stage, raw=s)

    if s.startswith("transcribing:"):
        try:
            pct = int(s[len("transcribing:"):])
        except ValueError:
            pct = 0
        return ParsedStatus("percent", percent=max(0, min(100, pct)), raw=s)

    if s.startswith("Done: "):
        return ParsedStatus("done_file", name=s[len("Done: "):], raw=s)

    if s == "Done.":
        return ParsedStatus("done_all", raw=s)

    if s == "Stopped":
        return ParsedStatus("stopped", raw=s)

    return ParsedStatus("other", raw=s)


class ProgressReporter:
    """Builds status strings and forwards them with an overall completion ratio."""

    def __init__(self, on_progress: ProgressCallback | None = None):
        self.on_progress = on_progress
        self.total = 0

    def start(self, total_files: int) -> None:
        self.total = max(0, total_files)

    # -- internals ---------------------------------------------------------
    def _emit(self, fraction: float, status: str) -> None:
        if self.on_progress is None:
            return
        self.on_progress(max(0.0, min(1.0, fraction)), status)

    def _overall(self, index: int, within_file: float = 0.0) -> float:
        if self.total <= 0:
            return 0.0
        return (index + max(0.0, min(1.0, within_file))) / self.total

    # -- public API --------------------------------------------------------
    def file_started(self, index: int, name: str) -> None:
        self._emit(self._overall(index), f"Processing {name}...")

    def stage(self, index: int, stage: str, name: str = "") -> None:
        self._emit(self._overall(index, 0.02), f"stage:{stage}:{name}")

    def file_percent(self, index: int, percent: int) -> None:
        pct = max(0, min(100, int(percent)))
        self._emit(self._overall(index, pct / 100.0), f"transcribing:{pct}")

    def file_done(self, index: int, name: str) -> None:
        """Must be called for every successfully processed file. See module docstring."""
        self._emit(self._overall(index, 1.0), f"Done: {name}")

    def stopped(self) -> None:
        self._emit(0.0, "Stopped")

    def all_done(self) -> None:
        self._emit(1.0, "Done.")
