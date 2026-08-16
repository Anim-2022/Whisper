"""Build the 'Settings' tab (Files / Model / Output sections).

The function takes the WhisperGUI instance and attaches widgets onto it
using the same attribute names the rest of the app already reads.
Behavior is identical to the original `WhisperGUI.build_settings_tab`;
this is a pure relocation + boilerplate-removal step.
"""
from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from .. import constants as C
from .field_factory import (
    make_field_label,
    make_help_label,
    make_labeled_browse_row,
    make_section_header,
)


def build(app) -> None:
    t = app.settings_frame
    t.grid_columnconfigure(1, weight=1)

    _build_intro_panel(app, t)
    _build_files_section(app, t)
    _build_model_section(app, t)
    _build_language_section(app, t)
    _build_output_section(app, t)


# ----------------------------------------------------------------------
# Intro panel — title, body, and three "step" pill labels
# ----------------------------------------------------------------------
def _build_intro_panel(app, parent) -> None:
    app.settings_intro = ctk.CTkFrame(
        parent,
        fg_color=C.COLOR_PANEL_BG,
        border_width=1,
        border_color=C.COLOR_PANEL_BORDER,
        corner_radius=18,
    )
    app.settings_intro.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 16))
    app.settings_intro.grid_columnconfigure((0, 1, 2), weight=1)

    app.settings_intro_title_label = ctk.CTkLabel(
        app.settings_intro,
        text=app.t("settings_intro_title"),
        font=ctk.CTkFont(size=C.SECTION_INTRO_TITLE_SIZE, weight="bold"),
    )
    app.settings_intro_title_label.grid(row=0, column=0, columnspan=3, padx=18, pady=(16, 4), sticky="w")

    app.settings_intro_body_label = ctk.CTkLabel(
        app.settings_intro,
        text=app.t("settings_intro_body"),
        text_color=C.COLOR_TEXT_INTRO_BODY,
        justify="left",
        wraplength=C.WRAPLENGTH_BODY,
    )
    app.settings_intro_body_label.grid(row=1, column=0, columnspan=3, padx=18, pady=(0, 12), sticky="w")

    app.settings_step_labels = []
    step_keys = ["settings_step_audio", "settings_step_model", "settings_step_start"]
    for idx, key in enumerate(step_keys):
        step_label = ctk.CTkLabel(
            app.settings_intro,
            text=app.t(key),
            fg_color=C.COLOR_STEP_BADGE_BG,
            corner_radius=999,
            padx=12,
            pady=6,
            text_color=C.COLOR_STEP_BADGE_TEXT,
        )
        step_label.grid(row=2, column=idx, padx=8, pady=(0, 16), sticky="w")
        app.settings_step_labels.append((step_label, key))


# ----------------------------------------------------------------------
# Files section — audio & output folder rows
# ----------------------------------------------------------------------
def _build_files_section(app, parent) -> None:
    app.files_section_label = make_section_header(app=app, parent=parent, row=1, text_key="section_files")

    app.audio_folder_label, app.entry_audio, app.btn_browse_audio = make_labeled_browse_row(
        parent=parent, app=app, row=2,
        label_key="label_audio_folder", help_key="help_audio_folder",
        on_browse=app.browse_audio,
        default=str(C.resolve_default_audio_dir(Path(__file__).resolve().parents[2])),
        placeholder_key="placeholder_audio_folder",
    )

    app.output_folder_label, app.entry_output, app.btn_browse_output = make_labeled_browse_row(
        parent=parent, app=app, row=4,
        label_key="label_output_folder", help_key="help_output_folder",
        on_browse=app.browse_output,
        default=str(C.resolve_default_output_dir(Path(__file__).resolve().parents[2])),
    )


# ----------------------------------------------------------------------
# Model section — combo + Refresh/Folder buttons + help texts
# ----------------------------------------------------------------------
def _build_model_section(app, parent) -> None:
    app.model_section_label = make_section_header(
        parent=parent, app=app, row=6, text_key="section_model", pady=(20, 5),
    )

    app.local_model_label = make_field_label(parent=parent, app=app, row=7, text_key="label_local_model")

    app.combo_model = ctk.CTkComboBox(
        parent,
        values=[C.DEFAULT_MODEL],
        height=34,
        command=lambda _value: app.refresh_runtime_summary(),
    )
    app.combo_model.grid(row=7, column=1, sticky="ew", padx=10, pady=5)
    app.combo_model.set(C.DEFAULT_MODEL)

    app.model_button_frame = ctk.CTkFrame(parent, fg_color="transparent")
    app.model_button_frame.grid(row=7, column=2, sticky="e", padx=10)
    app.btn_refresh_models = ctk.CTkButton(
        app.model_button_frame, text=app.t("button_refresh"),
        width=82, command=app.refresh_local_models,
    )
    app.btn_refresh_models.pack(side="left", padx=(0, 6))
    app.btn_model_folder = ctk.CTkButton(
        app.model_button_frame, text=app.t("button_folder"),
        width=82, command=app.browse_model_folder,
    )
    app.btn_model_folder.pack(side="left")

    app.model_help_label = ctk.CTkLabel(
        parent,
        text=app.describe_model(C.DEFAULT_MODEL),
        text_color=C.COLOR_TEXT_MODEL_HELP,
        justify="left",
        wraplength=C.WRAPLENGTH_BODY,
    )
    app.model_help_label.grid(row=8, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 4))

    app.local_models_note = ctk.CTkLabel(
        parent,
        text="",
        text_color=C.COLOR_TEXT_LOCAL_NOTE,
        justify="left",
        wraplength=C.WRAPLENGTH_BODY,
    )
    app.local_models_note.grid(row=9, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 2))

    make_help_label(parent=parent, app=app, row=10, text_key="help_model_buttons")


# ----------------------------------------------------------------------
# Transcription language combo + auto-detect checkbox
# ----------------------------------------------------------------------
def _build_language_section(app, parent) -> None:
    app.transcription_language_label = make_field_label(
        parent=parent, app=app, row=11, text_key="label_transcription_language",
    )
    app.combo_lang = ctk.CTkComboBox(parent, values=list(C.TRANSCRIPTION_LANGUAGES))
    app.combo_lang.grid(row=11, column=1, sticky="w", padx=10, pady=5)
    app.combo_lang.set(C.DEFAULTS["transcription_lang"])

    app.check_auto_lang = ctk.CTkCheckBox(parent, text=app.t("checkbox_auto_lang"))
    app.check_auto_lang.grid(row=11, column=2, sticky="w", padx=10, pady=5)

    make_help_label(parent=parent, app=app, row=12, text_key="help_language")


# ----------------------------------------------------------------------
# Output section — SRT toggle
# ----------------------------------------------------------------------
def _build_output_section(app, parent) -> None:
    app.output_section_label = make_section_header(
        parent=parent, app=app, row=13, text_key="section_output", pady=(20, 5),
    )
    app.check_srt = ctk.CTkCheckBox(parent, text=app.t("checkbox_srt"))
    app.check_srt.grid(row=14, column=1, sticky="w", padx=10, pady=(4, 5))
    make_help_label(parent=parent, app=app, row=15, text_key="help_srt")
