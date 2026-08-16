"""Small builders that eliminate label+entry+help-text boilerplate.

Each helper takes `app` (WhisperGUI) so it can:
  - call `app.t(key, **kwargs)` for translations
  - append created help labels to `app.localized_help_labels`, which the
    runtime language switcher iterates to retranslate.

Returned widgets are NOT auto-attached to `app` — the caller is
responsible for `app.<name> = widget` so that the rest of the app keeps
its public widget references and behavior is identical to the
pre-refactor code.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import customtkinter as ctk

from .. import constants as C
from .validated_entry import ValidatedEntry


def make_help_label(parent, app, row: int, text_key: str,
                    *, columnspan: int = 3, padx: int = 20):
    """Italic-style help line under a field. Tracked for re-localization."""
    label = ctk.CTkLabel(
        parent,
        text=app.t(text_key),
        text_color=C.COLOR_TEXT_HELP,
        font=ctk.CTkFont(size=C.HELP_FONT_SIZE),
        wraplength=C.WRAPLENGTH_BODY,
        justify="left",
    )
    label.grid(row=row, column=0, columnspan=columnspan,
               padx=padx, pady=(0, 6), sticky="w")
    app.localized_help_labels.append((label, text_key))
    return label


def make_section_header(parent, app, row: int, text_key: str,
                        *, pady=(0, 5), padx: int = 10):
    """Bold section header (e.g. '1. Files', 'Speed and compute')."""
    label = ctk.CTkLabel(
        parent,
        text=app.t(text_key),
        font=ctk.CTkFont(size=C.HEADER_FONT_SIZE, weight="bold"),
    )
    label.grid(row=row, column=0, sticky="w", padx=padx, pady=pady)
    return label


def make_field_label(parent, app, row: int, text_key: str):
    """Plain left-column label for a settings field."""
    label = ctk.CTkLabel(parent, text=app.t(text_key))
    label.grid(row=row, column=0, sticky="w", padx=20, pady=5)
    return label


def make_labeled_entry(parent, app, row: int, label_key: str,
                       help_key: str | None, *, default=None,
                       validator: Callable | None = None):
    """Label (col 0) + Entry (col 1), help line on `row + 1`.

    When `validator` is provided the widget is a `ValidatedEntry` that
    revalidates on every change and is registered into
    `app.validated_entries` so the pre-flight check in `start_process`
    can iterate every guarded field.

    Returns (label, entry). Help row is auto-created when `help_key` is set.
    """
    label = make_field_label(parent, app, row, label_key)
    if validator is not None:
        entry = ValidatedEntry(parent, validator=validator,
                               label_key=label_key, height=34)
        # The list is initialized in WhisperGUI.__init__ before tab build,
        # but be defensive in case a future builder runs in isolation.
        if not hasattr(app, "validated_entries"):
            app.validated_entries = []
        app.validated_entries.append(entry)
    else:
        entry = ctk.CTkEntry(parent, height=34)
    entry.grid(row=row, column=1, sticky="ew", padx=10, pady=5)
    if default is not None:
        entry.insert(0, str(default))
    if help_key:
        make_help_label(parent, app, row + 1, help_key)
    return label, entry


def make_labeled_combo(parent, app, row: int, label_key: str,
                       help_key: str | None, values: Sequence[str],
                       default: str | None = None,
                       *, command: Callable | None = None,
                       sticky: str = "ew"):
    """Label + ComboBox, help line on `row + 1`. Returns (label, combo)."""
    label = make_field_label(parent, app, row, label_key)
    combo = ctk.CTkComboBox(parent, values=list(values), height=34, command=command)
    combo.grid(row=row, column=1, sticky=sticky, padx=10, pady=5)
    if default is not None:
        combo.set(default)
    if help_key:
        make_help_label(parent, app, row + 1, help_key)
    return label, combo


def make_labeled_browse_row(parent, app, row: int, label_key: str,
                            help_key: str | None, on_browse: Callable,
                            *, default: str = "",
                            placeholder_key: str | None = None):
    """Label + Entry + 'Browse' button on one row, help line on next.

    Returns (label, entry, browse_button).
    """
    label = make_field_label(parent, app, row, label_key)
    if placeholder_key:
        entry = ctk.CTkEntry(parent, placeholder_text=app.t(placeholder_key), height=34)
    else:
        entry = ctk.CTkEntry(parent, height=34)
    entry.grid(row=row, column=1, sticky="ew", padx=10, pady=5)
    if default:
        entry.insert(0, default)
    btn = ctk.CTkButton(parent, text=app.t("button_browse"), width=86, command=on_browse)
    btn.grid(row=row, column=2, padx=20)
    if help_key:
        make_help_label(parent, app, row + 1, help_key)
    return label, entry, btn
