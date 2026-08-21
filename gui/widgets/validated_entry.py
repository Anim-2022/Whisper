"""`ValidatedEntry` — a CTkEntry that flags invalid input with a red border.

Split out of `gui.validators` so that module stays free of any customtkinter
import. The pure validators are exercised by the test suite on headless runners
where Tk is not installed at all, which is what makes CI on ubuntu-latest possible.
"""
from __future__ import annotations

import customtkinter as ctk

from ..validators import ValidationError

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
