"""Input validation for numeric Entry widgets.

`IntRangeValidator` / `FloatRangeValidator` / `PathValidator` are pure callables
that take the entry's current string and return a translation key + format kwargs
on error, or None when OK.

Validators return a tuple `(i18n_key, kwargs)` rather than a final
string so the GUI can render them in the user's chosen UI language.

Deliberately free of any customtkinter import: the widget that consumes these
lives in `gui.widgets.validated_entry`. That split is what lets the test suite
import this module on a headless runner with no Tk installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Type alias: an error is either None (valid) or (i18n_key, format_kwargs).
ValidationError = tuple[str, dict[str, Any]] | None


class IntRangeValidator:
    """Accepts integers within `[lo, hi]` inclusive.

    Returns the i18n key `validation_int_range` with `{lo, hi}` on failure.
    """

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi

    def __call__(self, value: str) -> ValidationError:
        try:
            n = int(str(value).strip())
        except (ValueError, TypeError):
            return ("validation_int_range", {"lo": self.lo, "hi": self.hi})
        if not (self.lo <= n <= self.hi):
            return ("validation_int_range", {"lo": self.lo, "hi": self.hi})
        return None


class FloatRangeValidator:
    """Accepts floats (or float-parseable ints) within `[lo, hi]` inclusive."""

    def __init__(self, lo: float, hi: float):
        self.lo = float(lo)
        self.hi = float(hi)

    def __call__(self, value: str) -> ValidationError:
        try:
            n = float(str(value).strip())
        except (ValueError, TypeError):
            return ("validation_float_range", {"lo": self.lo, "hi": self.hi})
        if not (self.lo <= n <= self.hi):
            return ("validation_float_range", {"lo": self.lo, "hi": self.hi})
        return None


class PathValidator:
    """Accepts a non-empty path string. Optionally requires existence."""

    def __init__(self, must_exist: bool = True):
        self.must_exist = must_exist

    def __call__(self, value: str) -> ValidationError:
        text = str(value).strip()
        if not text:
            return ("validation_path_empty", {})
        if self.must_exist and not Path(text).exists():
            return ("validation_path_missing", {"path": text})
        return None
