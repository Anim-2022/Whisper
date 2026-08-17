"""Build the 'Advanced' tab (Compute / VAD / Decoding sections).

Same conventions as `settings_tab.build`: attaches widgets onto the
WhisperGUI instance using exact attribute names that the rest of the
app reads.

Three sections rather than the previous two. The old "Segmentation" section
described a hand-rolled chunking loop (chunk length, overlap, merge gap, target
dBFS, token budget) that no longer exists — faster-whisper uses fixed 30 s
windows and merges speech by silence duration. What replaces it is the set of
knobs that actually change decoding quality.

Grid rows come from a shared `RowCounter` rather than being written out by hand:
each field also draws a help line on the row below it, so hand-numbering means a
single inserted field silently overlaps the next section's heading.
"""
from __future__ import annotations

import customtkinter as ctk

from .. import constants as C
from ..validators import FloatRangeValidator, IntRangeValidator
from .field_factory import (
    RowCounter,
    make_field_label,
    make_help_label,
    make_labeled_combo,
    make_labeled_entry,
    make_section_header,
)


def build(app) -> None:
    t = app.adv_frame
    t.grid_columnconfigure(1, weight=1)

    rows = RowCounter()
    _build_intro_panel(app, t, rows)
    _build_compute_section(app, t, rows)
    _build_vad_section(app, t, rows)
    _build_decoding_section(app, t, rows)


# ----------------------------------------------------------------------
# "Advanced = caution" intro panel
# ----------------------------------------------------------------------
def _build_intro_panel(app, parent, rows: RowCounter) -> None:
    app.adv_intro = ctk.CTkFrame(
        parent,
        fg_color=C.COLOR_ADV_PANEL_BG,
        border_width=1,
        border_color=C.COLOR_ADV_PANEL_BORDER,
        corner_radius=18,
    )
    app.adv_intro.grid(row=rows.take(), column=0, columnspan=3,
                       sticky="ew", padx=10, pady=(10, 16))

    app.adv_intro_title_label = ctk.CTkLabel(
        app.adv_intro,
        text=app.t("adv_intro_title"),
        font=ctk.CTkFont(size=C.SECTION_INTRO_TITLE_SIZE, weight="bold"),
    )
    app.adv_intro_title_label.grid(row=0, column=0, padx=18, pady=(16, 4), sticky="w")

    app.adv_intro_body_label = ctk.CTkLabel(
        app.adv_intro,
        text=app.t("adv_intro_body"),
        text_color=C.COLOR_TEXT_ADV_INTRO,
        justify="left",
        wraplength=C.WRAPLENGTH_BODY,
    )
    app.adv_intro_body_label.grid(row=1, column=0, padx=18, pady=(0, 16), sticky="w")


# ----------------------------------------------------------------------
# Compute: device, compute type, mode, batch size
# ----------------------------------------------------------------------
def _build_compute_section(app, parent, rows: RowCounter) -> None:
    app.compute_section_label = make_section_header(
        parent=parent, app=app, row=rows.take(), text_key="section_compute",
    )

    def refresh(_value):
        app.refresh_runtime_summary()

    app.device_label, app.combo_device = make_labeled_combo(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_device", help_key="help_device",
        values=C.DEVICES, default=None, command=refresh,
    )

    app.precision_label, app.combo_compute = make_labeled_combo(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_precision", help_key="help_precision",
        values=C.COMPUTE_TYPES, default=None, command=refresh,
    )

    app.mode_label_widget, app.combo_mode = make_labeled_combo(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_mode", help_key="help_mode",
        values=app.localized_mode_values(), default=None, command=refresh,
    )
    app.set_mode_selection("batched" if C.DEFAULTS["batched"] else "sequential")

    app.batch_size_label, app.entry_batch = make_labeled_entry(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_batch_size", help_key="help_batch_size",
        default=C.DEFAULTS["batch_size"],
        validator=IntRangeValidator(*C.BATCH_SIZE_RANGE),
    )


# ----------------------------------------------------------------------
# VAD: Silero is bundled with faster-whisper, so there is nothing to install
# and no "repo not found" hint any more.
# ----------------------------------------------------------------------
def _build_vad_section(app, parent, rows: RowCounter) -> None:
    app.vad_section_label = make_section_header(
        parent=parent, app=app, row=rows.take(), text_key="section_vad", pady=(20, 5),
    )

    checkbox_row = rows.take()
    app.use_vad_label = make_field_label(
        parent=parent, app=app, row=checkbox_row, text_key="label_use_vad",
    )
    app.check_vad = ctk.CTkCheckBox(parent, text=app.t("checkbox_enable_vad"))
    app.check_vad.grid(row=checkbox_row, column=1, sticky="w", padx=10, pady=5)
    app.check_vad.select()
    make_help_label(parent=parent, app=app, row=rows.take(), text_key="help_vad")

    # Threshold uses a slider + numeric readout, so it gets a custom row.
    slider_row = rows.take()
    app.vad_threshold_label = make_field_label(
        parent=parent, app=app, row=slider_row, text_key="label_vad_threshold",
    )
    app.slider_frame = ctk.CTkFrame(parent, fg_color="transparent")
    app.slider_frame.grid(row=slider_row, column=1, sticky="ew")
    app.slider_frame.grid_columnconfigure(0, weight=1)

    vad_lo, vad_hi = C.VAD_THRESHOLD_RANGE
    app.slider_vad = ctk.CTkSlider(
        app.slider_frame,
        from_=vad_lo, to=vad_hi,
        number_of_steps=8,
        command=app.update_vad_label,
    )
    app.slider_vad.grid(row=0, column=0, sticky="ew", padx=(10, 5), pady=5)
    app.slider_vad.set(C.DEFAULTS["vad_threshold"])

    app.lbl_vad_val = ctk.CTkLabel(
        app.slider_frame,
        text=f"{C.DEFAULTS['vad_threshold']:.2f}",
        width=40,
        font=ctk.CTkFont(weight="bold"),
    )
    app.lbl_vad_val.grid(row=0, column=1, padx=5)
    make_help_label(parent=parent, app=app, row=rows.take(), text_key="help_vad_threshold")

    app.min_speech_label, app.entry_vad_min_speech = make_labeled_entry(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_min_speech", help_key="help_min_speech",
        default=C.DEFAULTS["vad_min_speech_ms"],
        validator=IntRangeValidator(*C.VAD_MIN_SPEECH_MS_RANGE),
    )
    app.min_silence_label, app.entry_vad_silence = make_labeled_entry(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_min_silence", help_key="help_min_silence",
        default=C.DEFAULTS["vad_min_silence_ms"],
        validator=IntRangeValidator(*C.VAD_MIN_SILENCE_MS_RANGE),
    )
    app.speech_pad_label, app.entry_vad_pad = make_labeled_entry(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_speech_pad", help_key="help_speech_pad",
        default=C.DEFAULTS["vad_speech_pad_ms"],
        validator=IntRangeValidator(*C.VAD_SPEECH_PAD_MS_RANGE),
    )


# ----------------------------------------------------------------------
# Decoding: beam width and the guards against degenerate output
# ----------------------------------------------------------------------
def _build_decoding_section(app, parent, rows: RowCounter) -> None:
    app.decoding_section_label = make_section_header(
        parent=parent, app=app, row=rows.take(), text_key="section_decoding", pady=(20, 5),
    )

    app.beam_size_label, app.entry_beam_size = make_labeled_entry(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_beam_size", help_key="help_beam_size",
        default=C.DEFAULTS["beam_size"],
        validator=IntRangeValidator(*C.BEAM_SIZE_RANGE),
    )

    # Three checkboxes stacked in one column block. The first two only take
    # effect in sequential mode; TranscriptionConfig warns when they are set
    # alongside batched, so the UI never silently lies about them.
    app.check_temp_fallback = ctk.CTkCheckBox(parent, text=app.t("checkbox_temp_fallback"))
    app.check_temp_fallback.grid(row=rows.take(), column=1, sticky="w", padx=10, pady=(5, 0))
    app.check_condition_prev = ctk.CTkCheckBox(parent, text=app.t("checkbox_condition_prev"))
    app.check_condition_prev.grid(row=rows.take(), column=1, sticky="w", padx=10, pady=0)
    app.check_word_timestamps = ctk.CTkCheckBox(parent, text=app.t("checkbox_word_timestamps"))
    app.check_word_timestamps.grid(row=rows.take(), column=1, sticky="w", padx=10, pady=(0, 5))
    make_help_label(parent=parent, app=app, row=rows.take(), text_key="help_decode_flags")

    app.no_speech_label, app.entry_no_speech = make_labeled_entry(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_no_speech", help_key="help_no_speech",
        default=C.DEFAULTS["no_speech_threshold"],
        validator=FloatRangeValidator(*C.NO_SPEECH_THRESHOLD_RANGE),
    )
    app.compression_ratio_label, app.entry_compression_ratio = make_labeled_entry(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_compression_ratio", help_key="help_compression_ratio",
        default=C.DEFAULTS["compression_ratio_threshold"],
        validator=FloatRangeValidator(*C.COMPRESSION_RATIO_RANGE),
    )
    app.log_prob_label, app.entry_log_prob = make_labeled_entry(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_log_prob", help_key="help_log_prob",
        default=C.DEFAULTS["log_prob_threshold"],
        validator=FloatRangeValidator(*C.LOG_PROB_THRESHOLD_RANGE),
    )
    # Optional: blank means off, so it gets no range validator.
    app.hallucination_label, app.entry_hallucination = make_labeled_entry(
        parent=parent, app=app, row=rows.take(2),
        label_key="label_hallucination", help_key="help_hallucination",
        default=C.DEFAULTS["hallucination_silence_threshold"] or "",
    )
