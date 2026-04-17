# -*- coding: utf-8 -*-
"""Build the 'Advanced' tab (Speed&Compute / Segmentation sections).

Same conventions as `settings_tab.build`: attaches widgets onto the
WhisperGUI instance using exact attribute names that the rest of the
app reads. Behavior identical to the original `build_adv_tab`.
"""
from __future__ import annotations

import customtkinter as ctk

from .. import constants as C
from ..validators import IntRangeValidator, FloatRangeValidator
from .field_factory import (
    make_help_label,
    make_section_header,
    make_labeled_entry,
    make_labeled_combo,
    make_field_label,
)


def build(app) -> None:
    t = app.adv_frame
    t.grid_columnconfigure(1, weight=1)

    _build_intro_panel(app, t)
    _build_compute_section(app, t)
    _build_segmentation_section(app, t)


# ----------------------------------------------------------------------
# "Advanced = caution" intro panel
# ----------------------------------------------------------------------
def _build_intro_panel(app, parent) -> None:
    app.adv_intro = ctk.CTkFrame(
        parent,
        fg_color=C.COLOR_ADV_PANEL_BG,
        border_width=1,
        border_color=C.COLOR_ADV_PANEL_BORDER,
        corner_radius=18,
    )
    app.adv_intro.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 16))

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
# Speed & compute: device, precision, batch, VAD on/off + hint
# ----------------------------------------------------------------------
def _build_compute_section(app, parent) -> None:
    app.compute_section_label = make_section_header(
        parent=parent, app=app, row=1, text_key="section_compute",
    )

    refresh = lambda _value: app.refresh_runtime_summary()

    app.device_label, app.combo_device = make_labeled_combo(
        parent=parent, app=app, row=2,
        label_key="label_device", help_key="help_device",
        values=C.DEVICES, default=None, command=refresh,
    )

    app.precision_label, app.combo_dtype = make_labeled_combo(
        parent=parent, app=app, row=4,
        label_key="label_precision", help_key="help_precision",
        values=C.DTYPES, default=None, command=refresh,
    )

    app.batch_size_label, app.entry_batch = make_labeled_entry(
        parent=parent, app=app, row=6,
        label_key="label_batch_size", help_key="help_batch_size",
        default=C.DEFAULTS["batch_size"],
        validator=IntRangeValidator(*C.BATCH_SIZE_RANGE),
    )

    # VAD checkbox row — uses col 1 for the checkbox and col 2 for the
    # offline-repo hint label, so it doesn't fit the standard factory.
    app.use_vad_label = make_field_label(
        parent=parent, app=app, row=8, text_key="label_use_vad",
    )
    app.check_vad = ctk.CTkCheckBox(parent, text=app.t("checkbox_enable_vad"))
    app.check_vad.grid(row=8, column=1, sticky="w", padx=10, pady=5)
    app.check_vad.select()
    app.vad_hint_label = ctk.CTkLabel(
        parent,
        text=app.t("vad_hint_path", path="models\\silero-vad"),
        text_color=C.COLOR_TEXT_LOCAL_NOTE,
        justify="left",
        wraplength=C.WRAPLENGTH_VAD_HINT,
    )
    app.vad_hint_label.grid(row=8, column=2, sticky="w", padx=10, pady=5)
    make_help_label(parent=parent, app=app, row=9, text_key="help_vad")


# ----------------------------------------------------------------------
# Segmentation: VAD threshold slider + decode profile + numeric tunables
# ----------------------------------------------------------------------
def _build_segmentation_section(app, parent) -> None:
    app.segmentation_section_label = make_section_header(
        parent=parent, app=app, row=10,
        text_key="section_segmentation", pady=(20, 5),
    )

    # VAD threshold uses a slider + numeric label, so it gets a custom row
    # rather than a stock factory.
    app.vad_threshold_label = make_field_label(
        parent=parent, app=app, row=11, text_key="label_vad_threshold",
    )
    app.slider_frame = ctk.CTkFrame(parent, fg_color="transparent")
    app.slider_frame.grid(row=11, column=1, sticky="ew")
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
        text=f"{C.DEFAULTS['vad_threshold']:.1f}",
        width=40,
        font=ctk.CTkFont(weight="bold"),
    )
    app.lbl_vad_val.grid(row=0, column=1, padx=5)
    make_help_label(parent=parent, app=app, row=12, text_key="help_vad_threshold")

    # Decode profile combo (localized labels)
    app.decode_profile_label_widget, app.combo_decode = make_labeled_combo(
        parent=parent, app=app, row=13,
        label_key="label_decode_profile", help_key="help_decode_profile",
        values=app.localized_decode_profile_values(),
        default=None,
        command=lambda _value: app.refresh_runtime_summary(),
    )
    app.set_decode_profile_selection(C.DEFAULTS["decode_profile"])

    # Remaining numeric tunables — all label+entry+help triplets, each
    # guarded by the matching range from gui.constants so a typo can't
    # silently start a job that crashes inside whisper_core.
    app.target_db_label, app.entry_target_db = make_labeled_entry(
        parent=parent, app=app, row=15,
        label_key="label_target_db", help_key="help_target_db",
        default=C.DEFAULTS["target_db"],
        validator=FloatRangeValidator(*C.TARGET_DB_RANGE),
    )
    app.merge_gap_label, app.entry_vad_merge_gap = make_labeled_entry(
        parent=parent, app=app, row=17,
        label_key="label_merge_gap", help_key="help_merge_gap",
        default=C.DEFAULTS["vad_merge_gap"],
        validator=FloatRangeValidator(*C.VAD_MERGE_GAP_SEC_RANGE),
    )
    app.min_silence_label, app.entry_vad_silence = make_labeled_entry(
        parent=parent, app=app, row=19,
        label_key="label_min_silence", help_key="help_min_silence",
        default=C.DEFAULTS["vad_silence_ms"],
        validator=IntRangeValidator(*C.VAD_MIN_SILENCE_MS_RANGE),
    )
    app.chunk_sec_label, app.entry_chunk_sec = make_labeled_entry(
        parent=parent, app=app, row=21,
        label_key="label_chunk_sec", help_key="help_chunk_sec",
        default=C.DEFAULTS["chunk_sec"],
        validator=FloatRangeValidator(*C.CHUNK_SEC_RANGE),
    )
    app.overlap_sec_label, app.entry_overlap_sec = make_labeled_entry(
        parent=parent, app=app, row=23,
        label_key="label_overlap_sec", help_key="help_overlap_sec",
        default=C.DEFAULTS["overlap_sec"],
        validator=FloatRangeValidator(*C.OVERLAP_SEC_RANGE),
    )
    app.max_new_tokens_label, app.entry_max_tokens = make_labeled_entry(
        parent=parent, app=app, row=25,
        label_key="label_max_new_tokens", help_key="help_max_new_tokens",
        default=C.DEFAULTS["max_new_tokens"],
        validator=IntRangeValidator(*C.MAX_NEW_TOKENS_RANGE),
    )
