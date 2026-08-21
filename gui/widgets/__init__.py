"""Reusable widget builders for the WhisperGUI tabs.

Each `build(app)` function attaches widgets directly onto the `app`
(WhisperGUI instance) using exactly the attribute names that the rest of
the application reads — `app.entry_audio`, `app.combo_device`, etc.
This preserves the existing `apply_localization`, `apply_preset`,
`start_process`, and persistence code unchanged.

Modules:
    field_factory  — small helpers that remove the
                     "label + entry/combo + help-text" boilerplate
    settings_tab   — Files / Model / Output sections
    advanced_tab   — Speed&Compute / Segmentation
    logs_tab       — log intro panel + log textbox
"""
