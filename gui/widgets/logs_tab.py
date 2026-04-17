# -*- coding: utf-8 -*-
"""Build the 'Logs' tab (intro panel + log textbox).

Same conventions as the other tab modules. Behavior identical to the
original `WhisperGUI.build_log_tab`.
"""
from __future__ import annotations

import customtkinter as ctk

from .. import constants as C


def build(app) -> None:
    t = app.tab_logs
    t.grid_rowconfigure(1, weight=1)
    t.grid_columnconfigure(0, weight=1)

    app.logs_info_frame = ctk.CTkFrame(
        t,
        fg_color=C.COLOR_LOG_PANEL_BG,
        border_width=1,
        border_color=C.COLOR_PANEL_BORDER,
        corner_radius=16,
    )
    app.logs_info_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 10))

    app.logs_intro_title_label = ctk.CTkLabel(
        app.logs_info_frame,
        text=app.t("logs_intro_title"),
        font=ctk.CTkFont(size=C.HEADER_FONT_SIZE, weight="bold"),
    )
    app.logs_intro_title_label.grid(row=0, column=0, padx=16, pady=(14, 4), sticky="w")

    app.logs_intro_body_label = ctk.CTkLabel(
        app.logs_info_frame,
        text=app.t("logs_intro_body"),
        text_color=C.COLOR_TEXT_HINT,
        wraplength=C.WRAPLENGTH_LOGS_INTRO,
        justify="left",
    )
    app.logs_intro_body_label.grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")

    app.txt_log = ctk.CTkTextbox(t, state="disabled", font=("Consolas", 12))
    app.txt_log.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
