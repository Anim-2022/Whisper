"""Read-only preview window for the latest transcript file.

Lets the user inspect a finished transcript in any of the output formats
without opening an external editor.
Three actions: copy all, save as a copy, close. Failures (file vanished,
permission denied) are surfaced inline at the bottom of the window — we
never raise back to the caller.
"""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from .. import constants as C


class PreviewDialog(ctk.CTkToplevel):
    """Modal-ish Toplevel showing the contents of `path` in a CTkTextbox."""

    def __init__(self, master, path: Path, t):
        super().__init__(master)
        self._t = t
        self._path = Path(path)

        self.title(t("preview_title", name=self._path.name))
        self.geometry("780x560")
        self.minsize(520, 360)
        self.configure(fg_color=C.COLOR_BG)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._textbox = ctk.CTkTextbox(self, font=("Consolas", 12), wrap="word")
        self._textbox.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))

        self._status_label = ctk.CTkLabel(
            self, text="", text_color=C.COLOR_TEXT_HINT, anchor="w",
        )
        self._status_label.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 4))

        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        button_row.grid_columnconfigure(0, weight=1)

        self._btn_copy = ctk.CTkButton(
            button_row, text=t("preview_copy_all"), command=self._copy_all, width=130,
        )
        self._btn_copy.grid(row=0, column=1, padx=4)
        self._btn_save = ctk.CTkButton(
            button_row, text=t("preview_save_as"), command=self._save_as, width=130,
        )
        self._btn_save.grid(row=0, column=2, padx=4)
        self._btn_close = ctk.CTkButton(
            button_row, text=t("preview_close"), command=self.destroy, width=110,
        )
        self._btn_close.grid(row=0, column=3, padx=4)

        self._load_file()
        # Bring window to front + focus so the user actually notices it on
        # multi-monitor setups where Toplevels can hide behind the main window.
        self.after(50, self._raise_window)

    # ------------------------------------------------------------------
    def _load_file(self) -> None:
        try:
            text = self._path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            self._status_label.configure(
                text=self._t("preview_load_failed", error=e),
                text_color=C.COLOR_BUTTON_STOP_FG,
            )
            return
        self._textbox.delete("1.0", "end")
        self._textbox.insert("1.0", text)
        self._textbox.configure(state="disabled")

    def _raise_window(self) -> None:
        try:
            self.lift()
            self.focus_force()
        except Exception:
            pass

    def _copy_all(self) -> None:
        text = self._textbox.get("1.0", "end-1c")
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update_idletasks()
            self._status_label.configure(
                text=self._t("preview_copied"), text_color=C.COLOR_TEXT_HINT,
            )
        except Exception as e:
            self._status_label.configure(
                text=self._t("preview_load_failed", error=e),
                text_color=C.COLOR_BUTTON_STOP_FG,
            )

    def _save_as(self) -> None:
        # Default to whatever was opened, since the preview now handles every
        # output format rather than just plain text.
        target = filedialog.asksaveasfilename(
            parent=self,
            title=self._t("preview_save_dialog_title"),
            defaultextension=self._path.suffix or ".txt",
            initialfile=self._path.name,
            filetypes=[
                ("Text files", "*.txt"),
                ("Markdown", "*.md"),
                ("Subtitles", "*.srt *.vtt"),
                ("JSON", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not target:
            return
        try:
            Path(target).write_text(
                self._textbox.get("1.0", "end-1c"), encoding="utf-8"
            )
        except OSError as e:
            self._status_label.configure(
                text=self._t("preview_load_failed", error=e),
                text_color=C.COLOR_BUTTON_STOP_FG,
            )


def open_preview(master, path: Path | None, t) -> PreviewDialog | None:
    """Spawn a PreviewDialog if `path` exists; return the window (or None)."""
    if path is None or not Path(path).exists():
        return None
    return PreviewDialog(master, Path(path), t)
