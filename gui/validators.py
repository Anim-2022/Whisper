# -*- coding: utf-8 -*-
"""Input validation for numeric Entry widgets.

Two parts:
  * `IntRangeValidator` / `FloatRangeValidator` / `PathValidator` —
    pure callables that take the entry's current string and return
    a translation key + format kwargs on error, or None when OK.
  * `ValidatedEntry` — a CTkEntry subclass that runs the validator on
    every change (keyboard or programmatic), turns its border red on
    failure, and exposes `is_valid()` / `get_error()` for the
    pre-start-of-processing check.

Validators return a tuple `(i18n_key, kwargs)` rather than a final
string so the GUI can render them in the user's chosen UI language.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import customtkinter as ctk


# Type alias: an error is either None (valid) or (i18n_key, format_kwargs).
ValidationError = Optional[Tuple[str, Dict[str, Any]]]


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


# ----------------------------------------------------------------------
# Validated entry widget
# ----------------------------------------------------------------------
# Border color used to flag invalid input. Picked from the palette so it
# matches the existing Stop-button red and stays consistent in dark mode.
_INVALID_BORDER_COLOR = "#D63D3D"


class ValidatedEntry(ctk.CTkEntry):
    """CTkEntry that revalidates on every change and shows a red border on error.

    `label_key` is the i18n key of the field's label (e.g. `label_batch_size`).
    The pre-flight check in start_process uses it to build human-readable
    error messages like "Batch size: must be integer in [1, 32]".
    """

    def __init__(self, master, validator, *, label_key: str = "", **kwargs):
        super().__init__(master, **kwargs)
        self.validator = validator
        self.label_key = label_key
        # Capture whatever border color the theme assigned at construction
        # time so we can restore it after the user fixes the input.
        self._default_border = self.cget("border_color")
        self._invalid_border = _INVALID_BORDER_COLOR
        self._last_error: ValidationError = None
        # Keyboard typing fires <KeyRelease>; FocusOut is a safety net for
        # paste-and-tab-out flows that some IMEs swallow.
        self.bind("<KeyRelease>", self._revalidate_event)
        self.bind("<FocusOut>", self._revalidate_event)

    # CTkEntry.delete and .insert are how `set_entry_value` (used by
    # apply_preset) and the persistence loader mutate the field. Override
    # them so programmatic changes also fire validation — otherwise a
    # corrupt saved settings file would leave a stale red border or
    # (worse) a stale "valid" status.
    def insert(self, index, value):
        super().insert(index, value)
        self._revalidate()

    def delete(self, first_index, last_index=None):
        super().delete(first_index, last_index)
        self._revalidate()

    # ---- validation core ---------------------------------------------
    def _revalidate_event(self, _event=None):
        self._revalidate()

    def _revalidate(self) -> None:
        self._last_error = self.validator(self.get())
        if self._last_error is None:
            self.configure(border_color=self._default_border)
        else:
            self.configure(border_color=self._invalid_border)

    def is_valid(self) -> bool:
        """Run a fresh validation pass and return True iff the value is OK."""
        self._revalidate()
        return self._last_error is None

    def get_error(self) -> ValidationError:
        """Return the last `(i18n_key, kwargs)` error or None."""
        return self._last_error
