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
    RowCounter,
    make_field_label,
    make_help_label,
    make_labeled_browse_row,
    make_section_header,
)


def build(app) -> None:
    t = app.settings_frame
    t.grid_columnconfigure(1, weight=1)

    rows = RowCounter()
    _build_intro_panel(app, t, rows)
    _build_files_section(app, t, rows)
    _build_model_section(app, t, rows)
    _build_language_section(app, t, rows)
    _build_output_section(app, t, rows)


# ----------------------------------------------------------------------
# Intro panel — title, body, and three "step" pill labels
# ----------------------------------------------------------------------
def _build_intro_panel(app, parent, rows: RowCounter) -> None:
    app.settings_intro = ctk.CTkFrame(
        parent,
        fg_color=C.COLOR_PANEL_BG,
        border_width=1,
        border_color=C.COLOR_PANEL_BORDER,
        corner_radius=18,
    )
    app.settings_intro.grid(row=rows.take(), column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 16))
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
def _build_files_section(app, parent, rows: RowCounter) -> None:
    app.files_section_label = make_section_header(app=app, parent=parent, row=rows.take(), text_key="section_files")

    app.audio_folder_label, app.entry_audio, app.btn_browse_audio = make_labeled_browse_row(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_audio_folder", help_key="help_audio_folder",
        on_browse=app.browse_audio,
        default=str(C.resolve_default_audio_dir(Path(__file__).resolve().parents[2])),
        placeholder_key="placeholder_audio_folder",
    )

    app.output_folder_label, app.entry_output, app.btn_browse_output = make_labeled_browse_row(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_output_folder", help_key="help_output_folder",
        on_browse=app.browse_output,
        default=str(C.resolve_default_output_dir(Path(__file__).resolve().parents[2])),
    )


# ----------------------------------------------------------------------
# Model section — combo + Refresh/Folder buttons + help texts
# ----------------------------------------------------------------------
def _build_model_section(app, parent, rows: RowCounter) -> None:
    app.model_section_label = make_section_header(
        parent=parent, app=app, row=rows.take(), text_key="section_model", pady=(20, 5),
    )

    model_row = rows.take()
    app.local_model_label = make_field_label(parent=parent, app=app, row=model_row, text_key="label_local_model")

    app.combo_model = ctk.CTkComboBox(
        parent,
        values=[C.DEFAULT_MODEL],
        height=34,
        command=lambda _value: app.refresh_runtime_summary(),
    )
    app.combo_model.grid(row=model_row, column=1, sticky="ew", padx=10, pady=5)
    app.combo_model.set(C.DEFAULT_MODEL)

    app.model_button_frame = ctk.CTkFrame(parent, fg_color="transparent")
    app.model_button_frame.grid(row=model_row, column=2, sticky="e", padx=10)
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
    app.model_help_label.grid(row=rows.take(), column=0, columnspan=3, sticky="w", padx=20, pady=(0, 4))

    app.local_models_note = ctk.CTkLabel(
        parent,
        text="",
        text_color=C.COLOR_TEXT_LOCAL_NOTE,
        justify="left",
        wraplength=C.WRAPLENGTH_BODY,
    )
    app.local_models_note.grid(row=rows.take(), column=0, columnspan=3, sticky="w", padx=20, pady=(0, 2))

    make_help_label(parent=parent, app=app, row=rows.take(), text_key="help_model_buttons")


# ----------------------------------------------------------------------
# Transcription language combo + auto-detect checkbox
# ----------------------------------------------------------------------
def _build_language_section(app, parent, rows: RowCounter) -> None:
    lang_row = rows.take()
    app.transcription_language_label = make_field_label(
        parent=parent, app=app, row=lang_row, text_key="label_transcription_language",
    )
    app.combo_lang = ctk.CTkComboBox(parent, values=list(C.TRANSCRIPTION_LANGUAGES))
    app.combo_lang.grid(row=lang_row, column=1, sticky="w", padx=10, pady=5)
    app.combo_lang.set(C.DEFAULTS["transcription_lang"])

    app.check_auto_lang = ctk.CTkCheckBox(parent, text=app.t("checkbox_auto_lang"))
    app.check_auto_lang.grid(row=lang_row, column=2, sticky="w", padx=10, pady=5)

    make_help_label(parent=parent, app=app, row=rows.take(), text_key="help_language")

    # Seeding the decoder with attendee names and product jargon measurably
    # improves how they are spelled, so it belongs next to the language choice
    # rather than buried in Advanced.
    prompt_row = rows.take()
    app.initial_prompt_label = make_field_label(
        parent=parent, app=app, row=prompt_row, text_key="label_initial_prompt",
    )
    app.entry_initial_prompt = ctk.CTkEntry(
        parent, placeholder_text=app.t("placeholder_initial_prompt"),
    )
    app.entry_initial_prompt.grid(row=prompt_row, column=1, columnspan=2, sticky="ew", padx=10, pady=5)
    make_help_label(parent=parent, app=app, row=rows.take(), text_key="help_initial_prompt")


# ----------------------------------------------------------------------
# Output section — one checkbox per format
# ----------------------------------------------------------------------
def _build_output_section(app, parent, rows: RowCounter) -> None:
    app.output_section_label = make_section_header(
        parent=parent, app=app, row=rows.take(), text_key="section_output", pady=(20, 5),
    )

    app.format_frame = ctk.CTkFrame(parent, fg_color="transparent")
    app.format_frame.grid(row=rows.take(), column=1, columnspan=2, sticky="w", padx=10, pady=(4, 5))

    app.format_checkboxes = {}
    for column, fmt in enumerate(C.OUTPUT_FORMATS):
        checkbox = ctk.CTkCheckBox(app.format_frame, text=app.t(f"checkbox_format_{fmt}"))
        checkbox.grid(row=0, column=column, sticky="w", padx=(0, 14))
        if fmt in C.DEFAULT_OUTPUT_FORMATS:
            checkbox.select()
        app.format_checkboxes[fmt] = checkbox

    make_help_label(parent=parent, app=app, row=rows.take(), text_key="help_formats")
