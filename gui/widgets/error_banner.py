"""A dismissable banner shown above the tabview for errors and warnings.

Goal: surface problems (validation, worker crashes, missing folders)
without forcing the user to open the Logs tab. Hidden by default via
`grid_remove`; `show(level, message)` reinstates it with colors matching
`level`.

  level="error"  -> red background (matches the Stop-button palette)
  level="warn"   -> amber background (matches the "advanced" intro color)

The banner has one close button ("×") wired to `hide()`.
"""
from __future__ import annotations

import customtkinter as ctk

from .. import constants as C

# Dark-mode friendly pairs. Keeping them local (not in constants.py) since
# they're only meaningful to the banner widget itself.
_ERROR_BG = "#3B1A1A"
_ERROR_BORDER = C.COLOR_BUTTON_STOP_FG   # "#D63D3D"
_ERROR_TEXT = "#F5D6D6"
_WARN_BG = "#2F2713"
_WARN_BORDER = C.COLOR_TEXT_ADV_INTRO    # "#D7B16E"
_WARN_TEXT = "#F0DFB8"


class ErrorBanner(ctk.CTkFrame):
    """Tall single-row banner with an icon, a multi-line message, and ✕.

    Attach the banner with `grid(...)` once, then call `hide()` so it
    starts invisible. Every `show(level, message)` restores the widget
    with the new level's colors.
    """

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            fg_color=_ERROR_BG,
            border_width=1,
            border_color=_ERROR_BORDER,
            corner_radius=10,
            **kwargs,
        )
        # Two columns: flexible message area + fixed close button.
        self.grid_columnconfigure(1, weight=1)

        self._icon_label = ctk.CTkLabel(
            self, text="!", width=28,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=_ERROR_BORDER,
        )
        self._icon_label.grid(row=0, column=0, padx=(12, 6), pady=8, sticky="nw")

        self._message_label = ctk.CTkLabel(
            self, text="",
            text_color=_ERROR_TEXT,
            justify="left",
            wraplength=900,
            anchor="w",
        )
        self._message_label.grid(row=0, column=1, padx=(0, 6), pady=8, sticky="ew")

        self._close_button = ctk.CTkButton(
            self, text="×", width=30, height=30,
            fg_color="transparent", hover_color=_ERROR_BORDER,
            text_color=_ERROR_TEXT,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.hide,
        )
        self._close_button.grid(row=0, column=2, padx=(0, 8), pady=6, sticky="ne")

        self._visible = False

    # ------------------------------------------------------------------
    def show(self, message: str, level: str = "error") -> None:
        """Make the banner visible with the given message and level colors."""
        if level == "warn":
            bg, border, text = _WARN_BG, _WARN_BORDER, _WARN_TEXT
        else:
            bg, border, text = _ERROR_BG, _ERROR_BORDER, _ERROR_TEXT

        self.configure(fg_color=bg, border_color=border)
        self._icon_label.configure(text_color=border)
        self._message_label.configure(text=message, text_color=text)
        self._close_button.configure(text_color=text, hover_color=border)

        if not self._visible:
            self.grid()           # re-show if previously grid_remove'd
            self._visible = True

    def hide(self) -> None:
        """Remove the banner from the layout (keeps widget alive for reuse).

        Unconditional grid_remove — the caller typically does
        `banner.grid(...); banner.hide()` during construction, when
        self._visible is still False even though the widget is already
        shown. The old visibility guard skipped grid_remove in that
        case and left an empty banner on screen at launch.
        """
        self.grid_remove()
        self._visible = False

    def is_visible(self) -> bool:
        return self._visible
