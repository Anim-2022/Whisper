import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

try:
    from whisper_core import TranscriptionConfig, WhisperCore
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from whisper_core import TranscriptionConfig, WhisperCore

# Single source of truth for theme/colors/sizes/lists/presets lives in gui/.
from gui import constants as C
from gui import settings_store
from gui.i18n import UI_TEXT
from gui.widgets import advanced_tab as advanced_tab_builder
from gui.widgets import logs_tab as logs_tab_builder
from gui.widgets import settings_tab as settings_tab_builder
from gui.widgets.error_banner import ErrorBanner
from gui.widgets.preview_dialog import open_preview
from whisper_engine.audio import find_audio_files
from whisper_engine.device import has_cuda
from whisper_engine.models import discover_ct2_models
from whisper_engine.progress import parse_status
from whisper_engine.writers import pick_preview

# Optional drag-and-drop support. windnd is Windows-only and tiny; if it
# isn't installed we silently skip D&D wiring rather than fail to start.
try:
    import windnd  # type: ignore
    _HAS_WINDND = True
except Exception:
    windnd = None  # type: ignore
    _HAS_WINDND = False


ctk.set_appearance_mode(C.APPEARANCE_MODE)
ctk.set_default_color_theme(C.COLOR_THEME)


class WhisperGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(C.WINDOW_TITLE)
        self.geometry(C.WINDOW_GEOMETRY)
        self.minsize(*C.WINDOW_MIN_SIZE)
        self.configure(fg_color=C.COLOR_BG)

        self.core = WhisperCore(
            on_log=self.on_core_log,
            on_progress=self.on_core_progress,
            on_result=self.on_core_result,
        )
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        # Banner messages enqueued from the worker thread; drained on the
        # Tk main thread inside process_queues. Items: (level, message).
        self.banner_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        # Written files reported by the core. Items: (audio_path, [written paths]).
        self.result_queue: queue.Queue[tuple[Path, list[Path]]] = queue.Queue()
        self.is_running = False
        self.files_to_process: list[Path] = []
        self.ui_language = C.DEFAULT_UI_LANGUAGE
        self.preset_keys = list(C.PRESET_KEYS)
        self.mode_keys = list(C.MODE_KEYS)
        self.localized_help_labels = []
        # Filled by field_factory.make_labeled_entry whenever a validator is
        # passed. start_process iterates this list as a pre-flight check.
        self.validated_entries = []
        self.current_status_raw = "Ready"
        # Per-file progress tracking (Phase E). Reset on each start_process.
        self.files_total = 0
        self.current_file_index = 0
        self.current_file_name = ""
        self._file_started_at = 0.0
        self._completed_durations: list[float] = []
        # Probed via CTranslate2 rather than torch, which is no longer a dependency.
        self.has_cuda = has_cuda()
        # Phase F: post-run "Open folder" / "View result" need to know what was
        # produced. Updated whenever we see a "Done: <name>" status from core.
        self.last_output_dir: Path | None = None
        self.last_result_path: Path | None = None
        # Widgets that should grey out while a job is running. Filled below as
        # the sidebar / tabs are built.
        self._input_widgets: list = []

        self.grid_columnconfigure(1, weight=1)
        # row 0 (banner) stays its natural height; row 1 (tabview) takes
        # all remaining vertical space. The sidebar spans both rows.
        self.grid_rowconfigure(1, weight=1)

        self.sidebar_frame = ctk.CTkFrame(
            self, width=C.SIDEBAR_WIDTH, corner_radius=0, fg_color=C.COLOR_SIDEBAR_BG,
        )
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Whisper\nUnified",
            font=ctk.CTkFont(size=C.LOGO_FONT_SIZE, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(28, 6), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=self.t("subtitle"),
            text_color=C.COLOR_TEXT_SUBTITLE,
            font=ctk.CTkFont(size=C.HELP_FONT_SIZE),
            justify="left",
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 18), sticky="w")

        self.ui_language_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=self.t("label_interface_language"),
            font=ctk.CTkFont(weight="bold"),
        )
        self.ui_language_label.grid(row=2, column=0, padx=20, sticky="w")
        self.combo_ui_lang = ctk.CTkComboBox(
            self.sidebar_frame,
            values=[self.ui_language_label_text("ru"), self.ui_language_label_text("en")],
            command=self.change_ui_language,
            height=34,
        )
        self.combo_ui_lang.grid(row=3, column=0, padx=20, pady=(0, 14), sticky="ew")
        self.combo_ui_lang.set(self.ui_language_label_text(self.ui_language))

        self.preset_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=self.t("label_preset"),
            font=ctk.CTkFont(weight="bold"),
        )
        self.preset_label.grid(row=4, column=0, padx=20, sticky="w")
        self.combo_preset = ctk.CTkComboBox(
            self.sidebar_frame,
            values=self.localized_preset_values(),
            command=self.apply_preset,
            height=34,
        )
        self.combo_preset.grid(row=5, column=0, padx=20, pady=(0, 8), sticky="ew")

        self.preset_summary_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="",
            text_color=C.COLOR_TEXT_PRESET_SUMMARY,
            font=ctk.CTkFont(size=C.HELP_FONT_SIZE),
            wraplength=C.WRAPLENGTH_SIDEBAR,
            justify="left",
        )
        self.preset_summary_label.grid(row=6, column=0, padx=20, pady=(0, 14), sticky="w")

        self.runtime_summary_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="",
            text_color=C.COLOR_TEXT_RUNTIME,
            font=ctk.CTkFont(size=C.HELP_FONT_SIZE),
            wraplength=C.WRAPLENGTH_SIDEBAR,
            justify="left",
        )
        self.runtime_summary_label.grid(row=7, column=0, padx=20, pady=(0, 16), sticky="w")

        self.btn_start = ctk.CTkButton(
            self.sidebar_frame,
            text=self.t("button_start"),
            command=self.start_process,
            fg_color=C.COLOR_BUTTON_START_FG,
            hover_color=C.COLOR_BUTTON_START_HOVER,
            text_color=C.COLOR_BUTTON_START_TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
        )
        self.btn_start.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.btn_stop = ctk.CTkButton(
            self.sidebar_frame,
            text=self.t("button_stop"),
            command=self.stop_process,
            fg_color=C.COLOR_BUTTON_STOP_FG,
            hover_color=C.COLOR_BUTTON_STOP_HOVER,
            state="disabled",
            height=36,
        )
        self.btn_stop.grid(row=9, column=0, padx=20, pady=(0, 14), sticky="ew")

        # Phase F: post-run actions. Disabled until the worker reports a
        # finished file; enabled by _set_post_run_actions(True).
        self.btn_open_folder = ctk.CTkButton(
            self.sidebar_frame,
            text=self.t("button_open_results"),
            command=self.open_results_folder,
            height=32,
            state="disabled",
        )
        self.btn_open_folder.grid(row=14, column=0, padx=20, pady=(0, 6), sticky="ew")

        self.btn_view_result = ctk.CTkButton(
            self.sidebar_frame,
            text=self.t("button_view_result"),
            command=self.preview_latest_result,
            height=32,
            state="disabled",
        )
        self.btn_view_result.grid(row=15, column=0, padx=20, pady=(0, 12), sticky="ew")

        self.sidebar_hint_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=self.t("sidebar_hint"),
            text_color=C.COLOR_TEXT_HINT,
            font=ctk.CTkFont(size=C.HELP_FONT_SIZE),
            wraplength=C.WRAPLENGTH_SIDEBAR,
            justify="left",
        )
        self.sidebar_hint_label.grid(row=10, column=0, padx=20, pady=(0, 18), sticky="w")

        self.hotkey_hint_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=self.t("hotkey_hint"),
            text_color=C.COLOR_TEXT_HINT,
            font=ctk.CTkFont(size=C.HELP_FONT_SIZE),
            wraplength=C.WRAPLENGTH_SIDEBAR,
            justify="left",
        )
        self.hotkey_hint_label.grid(row=16, column=0, padx=20, pady=(8, 14), sticky="w")

        self.progress_bar = ctk.CTkProgressBar(self.sidebar_frame, height=12)
        self.progress_bar.grid(row=11, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.progress_bar.set(0)

        self.lbl_percentage = ctk.CTkLabel(
            self.sidebar_frame,
            text="0%",
            font=ctk.CTkFont(size=C.HELP_FONT_SIZE, weight="bold"),
        )
        self.lbl_percentage.grid(row=12, column=0, padx=20, pady=(4, 0), sticky="w")

        self.status_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=self.t("status_ready"),
            wraplength=C.WRAPLENGTH_SIDEBAR,
            text_color=C.COLOR_TEXT_STATUS,
            justify="left",
        )
        self.status_label.grid(row=13, column=0, padx=20, pady=(10, 20), sticky="w")

        self.tabview = ctk.CTkTabview(
            self,
            fg_color=C.COLOR_BG,
            segmented_button_fg_color=C.COLOR_TAB_BG,
            segmented_button_selected_color=C.COLOR_TAB_SELECTED,
            segmented_button_selected_hover_color=C.COLOR_TAB_SELECTED_HOVER,
        )
        # Error/warning banner sits above the tabview so problems are
        # visible no matter which tab is open. Hidden until show() is called.
        self.error_banner = ErrorBanner(self)
        self.error_banner.grid(row=0, column=1, padx=20, pady=(20, 0), sticky="ew")
        self.error_banner.hide()

        self.tabview.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

        self.tab_names = {
            "settings": self.t("tab_settings"),
            "advanced": self.t("tab_advanced"),
            "logs": self.t("tab_logs"),
        }
        self.tab_settings = self.tabview.add(self.tab_names["settings"])
        self.tab_adv = self.tabview.add(self.tab_names["advanced"])
        self.tab_logs = self.tabview.add(self.tab_names["logs"])

        self.settings_frame = ctk.CTkScrollableFrame(self.tab_settings, fg_color="transparent")
        self.settings_frame.pack(fill="both", expand=True)
        self.adv_frame = ctk.CTkScrollableFrame(self.tab_adv, fg_color="transparent")
        self.adv_frame.pack(fill="both", expand=True)

        # Tab UI is built by the gui.widgets.* modules. They attach widgets
        # directly onto self using the same attribute names the rest of the
        # app reads (entry_audio, combo_device, etc.), so apply_localization,
        # apply_preset, start_process, and persistence keep working unchanged.
        settings_tab_builder.build(self)
        advanced_tab_builder.build(self)
        logs_tab_builder.build(self)

        self.check_hardware()
        self.refresh_local_models()
        self.set_preset_selection(C.DEFAULT_PRESET)
        self.apply_preset(C.DEFAULT_PRESET)

        # Load persisted user settings (window geometry, last preset, paths,
        # tweaked numeric fields). This must run AFTER the default preset is
        # applied so individual saved field values override the preset baseline.
        self._load_persisted_settings()

        # Persist current values when the user closes the window.
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Phase F: hotkeys + drag-and-drop. Bind globally so they work no
        # matter which tab/widget has focus.
        self._bind_hotkeys()
        self._wire_drag_and_drop()
        # Build the list of widgets to grey out while a job is running.
        self._collect_input_widgets()

        self.after(100, self.process_queues)

    def t(self, key: str, **kwargs) -> str:
        language_map = UI_TEXT.get(self.ui_language, UI_TEXT["en"])
        text = language_map.get(key, UI_TEXT["en"].get(key, key))
        return text.format(**kwargs)

    def ui_language_label_text(self, language_code: str) -> str:
        return {"ru": "Русский", "en": "English"}.get(language_code, "English")

    def ui_language_from_value(self, value: str) -> str:
        return "ru" if value.strip().lower().startswith("рус") else "en"

    def preset_display_text(self, preset_key: str) -> str:
        return self.t(f"preset_{preset_key}")

    def preset_description(self, preset_key: str) -> str:
        return self.t(f"preset_{preset_key}_desc")

    def localized_preset_values(self) -> list[str]:
        return [self.preset_display_text(key) for key in self.preset_keys]

    def preset_key_from_value(self, value: str) -> str:
        normalized = value.strip()
        for language_code in UI_TEXT:
            for preset_key in self.preset_keys:
                if normalized == UI_TEXT[language_code][f"preset_{preset_key}"]:
                    return preset_key
        return C.DEFAULT_PRESET

    def selected_preset_key(self) -> str:
        return self.preset_key_from_value(self.combo_preset.get()) if hasattr(self, "combo_preset") else C.DEFAULT_PRESET

    def set_preset_selection(self, preset_key: str):
        if hasattr(self, "combo_preset"):
            self.combo_preset.set(self.preset_display_text(preset_key))

    def mode_label(self, mode_key: str) -> str:
        return self.t(f"mode_{mode_key}")

    def localized_mode_values(self) -> list[str]:
        return [self.mode_label(key) for key in self.mode_keys]

    def mode_key_from_value(self, value: str) -> str:
        normalized = value.strip()
        for language_code in UI_TEXT:
            for mode_key in self.mode_keys:
                if normalized == UI_TEXT[language_code][f"mode_{mode_key}"]:
                    return mode_key
        return "batched"

    def selected_mode_key(self) -> str:
        return self.mode_key_from_value(self.combo_mode.get()) if hasattr(self, "combo_mode") else "batched"

    def set_mode_selection(self, mode_key: str):
        if hasattr(self, "combo_mode"):
            self.combo_mode.set(self.mode_label(mode_key))

    def current_tab_key(self) -> str:
        current_name = self.tabview.get()
        for key, value in self.tab_names.items():
            if value == current_name:
                return key
        return "settings"

    def change_ui_language(self, value: str):
        new_language = self.ui_language_from_value(value)
        if new_language == self.ui_language:
            return

        self.ui_language = new_language
        self.apply_localization()

    def apply_localization(self):
        current_tab = self.current_tab_key()
        current_preset = self.selected_preset_key()
        current_mode = self.selected_mode_key()

        new_tab_names = {
            "settings": self.t("tab_settings"),
            "advanced": self.t("tab_advanced"),
            "logs": self.t("tab_logs"),
        }
        for key, old_name in list(self.tab_names.items()):
            new_name = new_tab_names[key]
            if old_name != new_name:
                self.tabview.rename(old_name, new_name)
        self.tab_names = new_tab_names
        self.tabview.set(self.tab_names[current_tab])

        self.subtitle_label.configure(text=self.t("subtitle"))
        self.ui_language_label.configure(text=self.t("label_interface_language"))
        self.combo_ui_lang.configure(values=[self.ui_language_label_text("ru"), self.ui_language_label_text("en")])
        self.combo_ui_lang.set(self.ui_language_label_text(self.ui_language))
        self.preset_label.configure(text=self.t("label_preset"))
        self.combo_preset.configure(values=self.localized_preset_values())
        self.set_preset_selection(current_preset)
        self.btn_start.configure(text=self.t("button_start"))
        self.btn_stop.configure(text=self.t("button_stop"))
        self.sidebar_hint_label.configure(text=self.t("sidebar_hint"))
        if hasattr(self, "hotkey_hint_label"):
            self.hotkey_hint_label.configure(text=self.t("hotkey_hint"))
        if hasattr(self, "btn_open_folder"):
            self.btn_open_folder.configure(text=self.t("button_open_results"))
        if hasattr(self, "btn_view_result"):
            self.btn_view_result.configure(text=self.t("button_view_result"))

        self.settings_intro_title_label.configure(text=self.t("settings_intro_title"))
        self.settings_intro_body_label.configure(text=self.t("settings_intro_body"))
        for label, key in self.settings_step_labels:
            label.configure(text=self.t(key))
        self.files_section_label.configure(text=self.t("section_files"))
        self.audio_folder_label.configure(text=self.t("label_audio_folder"))
        self.entry_audio.configure(placeholder_text=self.t("placeholder_audio_folder"))
        self.btn_browse_audio.configure(text=self.t("button_browse"))
        self.output_folder_label.configure(text=self.t("label_output_folder"))
        self.btn_browse_output.configure(text=self.t("button_browse"))
        self.model_section_label.configure(text=self.t("section_model"))
        self.local_model_label.configure(text=self.t("label_local_model"))
        self.btn_refresh_models.configure(text=self.t("button_refresh"))
        self.btn_model_folder.configure(text=self.t("button_folder"))
        self.transcription_language_label.configure(text=self.t("label_transcription_language"))
        self.check_auto_lang.configure(text=self.t("checkbox_auto_lang"))
        self.initial_prompt_label.configure(text=self.t("label_initial_prompt"))
        self.entry_initial_prompt.configure(placeholder_text=self.t("placeholder_initial_prompt"))
        self.output_section_label.configure(text=self.t("section_output"))
        for fmt, checkbox in self.format_checkboxes.items():
            checkbox.configure(text=self.t(f"checkbox_format_{fmt}"))

        self.adv_intro_title_label.configure(text=self.t("adv_intro_title"))
        self.adv_intro_body_label.configure(text=self.t("adv_intro_body"))
        self.compute_section_label.configure(text=self.t("section_compute"))
        self.device_label.configure(text=self.t("label_device"))
        self.precision_label.configure(text=self.t("label_precision"))
        self.batch_size_label.configure(text=self.t("label_batch_size"))
        self.mode_label_widget.configure(text=self.t("label_mode"))
        self.combo_mode.configure(values=self.localized_mode_values())
        self.set_mode_selection(current_mode)

        self.vad_section_label.configure(text=self.t("section_vad"))
        self.check_vad.configure(text=self.t("checkbox_enable_vad"))
        self.use_vad_label.configure(text=self.t("label_use_vad"))
        self.vad_threshold_label.configure(text=self.t("label_vad_threshold"))
        self.min_speech_label.configure(text=self.t("label_min_speech"))
        self.min_silence_label.configure(text=self.t("label_min_silence"))
        self.speech_pad_label.configure(text=self.t("label_speech_pad"))

        self.decoding_section_label.configure(text=self.t("section_decoding"))
        self.beam_size_label.configure(text=self.t("label_beam_size"))
        self.check_temp_fallback.configure(text=self.t("checkbox_temp_fallback"))
        self.check_condition_prev.configure(text=self.t("checkbox_condition_prev"))
        self.check_word_timestamps.configure(text=self.t("checkbox_word_timestamps"))
        self.no_speech_label.configure(text=self.t("label_no_speech"))
        self.compression_ratio_label.configure(text=self.t("label_compression_ratio"))
        self.log_prob_label.configure(text=self.t("label_log_prob"))
        self.hallucination_label.configure(text=self.t("label_hallucination"))

        self.logs_intro_title_label.configure(text=self.t("logs_intro_title"))
        self.logs_intro_body_label.configure(text=self.t("logs_intro_body"))

        for label, key in self.localized_help_labels:
            label.configure(text=self.t(key))

        self.refresh_local_models()
        self.status_label.configure(text=self.translate_status(self.current_status_raw))

    def translate_status(self, status: str) -> str:
        clean_status = status.strip()
        if clean_status in {"Done", "Done."}:
            return self.t("status_done")
        if clean_status == "Ready":
            return self.t("status_ready")
        if clean_status == "Error":
            return self.t("status_error")
        if "Stopped" in clean_status:
            return self.t("status_stopped")
        parsed = parse_status(status)
        if parsed.kind == "processing":
            return f"{self.t('status_processing_file_prefix')}{parsed.name}"
        if parsed.kind == "percent":
            return self.t("status_processing_percent", percent=parsed.percent)
        if parsed.kind == "stage":
            # Long files spend up to a minute decoding and running VAD before the
            # first segment arrives; without this the UI looks frozen.
            key = f"status_stage_{parsed.stage}"
            if key in UI_TEXT[self.ui_language]:
                return self.t(key)
            return clean_status
        return clean_status

    def check_hardware(self):
        self.has_cuda = has_cuda()
        if not self.has_cuda:
            messagebox.showwarning(
                self.t("hardware_check_title"),
                self.t("hardware_check_body"),
            )
        self.refresh_runtime_summary()

    def short_model_name(self, model_value: str) -> str:
        if Path(model_value).exists():
            return Path(model_value).name
        return model_value.split("/")[-1]

    def describe_model(self, model_value: str) -> str:
        lower = model_value.lower()
        if Path(model_value).exists():
            return self.t("model_desc_custom")
        # Checked before "large" because the name contains it, and the two behave
        # very differently: turbo is distilled and trades punctuation for speed.
        if "turbo" in lower:
            return self.t("model_desc_turbo")
        if "medium" in lower:
            return self.t("model_desc_medium")
        if "small" in lower:
            return self.t("model_desc_small")
        if "large" in lower:
            return self.t("model_desc_large")
        return self.t("model_desc_local_generic")

    def refresh_runtime_summary(self):
        preset = self.selected_preset_key()
        model_value = self.combo_model.get().strip() if hasattr(self, "combo_model") else C.DEFAULT_MODEL

        self.preset_summary_label.configure(text=self.preset_description(preset))

        compute_value = self.combo_device.get() if hasattr(self, "combo_device") else ("cuda" if self.has_cuda else "cpu")
        mode_key = self.selected_mode_key()
        runtime_lines = [
            self.t("runtime_model_line", model=self.short_model_name(model_value)),
            self.t("runtime_compute_line", compute=compute_value.upper()),
            self.t("runtime_mode_line", mode=self.mode_label(mode_key)),
            self.t("runtime_formats_line", formats=", ".join(self.selected_formats()) or "—"),
        ]
        self.runtime_summary_label.configure(text="\n".join(runtime_lines))

        if hasattr(self, "model_help_label"):
            self.model_help_label.configure(text=self.describe_model(model_value))

    def set_entry_value(self, entry: ctk.CTkEntry, value):
        entry.delete(0, "end")
        entry.insert(0, "" if value is None else str(value))

    def discover_local_models(self) -> list[str]:
        """Converted CTranslate2 models under models/ct2/."""
        return discover_ct2_models()

    def refresh_local_models(self):
        choices = self.discover_local_models()
        if not choices:
            choices = [C.DEFAULT_MODEL]

        current_value = self.combo_model.get().strip() if hasattr(self, "combo_model") else ""
        if current_value and current_value not in choices and Path(current_value).exists():
            choices.append(current_value)
        self.combo_model.configure(values=choices)

        if current_value in choices:
            self.combo_model.set(current_value)
        elif C.DEFAULT_MODEL in choices:
            self.combo_model.set(C.DEFAULT_MODEL)
        else:
            self.combo_model.set(choices[0])

        if hasattr(self, "local_models_note"):
            readable = ", ".join(self.short_model_name(choice) for choice in choices)
            self.local_models_note.configure(text=self.t("detected_local_models", models=readable))
        self.refresh_runtime_summary()

    def browse_model_folder(self):
        directory = filedialog.askdirectory()
        if not directory:
            return

        current_choices = list(self.combo_model.cget("values"))
        if directory not in current_choices:
            current_choices.append(directory)
            self.combo_model.configure(values=current_choices)
        self.combo_model.set(directory)
        self.refresh_runtime_summary()

    def apply_preset(self, choice):
        """Apply preset values from C.PRESETS to the GUI fields.

        All numeric data lives in gui.constants.PRESETS. This method only
        translates a preset key into widget calls; it never decides values
        itself. To tune a preset, edit constants.PRESETS — not this code.
        """
        preset_key = choice if choice in self.preset_keys else self.preset_key_from_value(choice)
        preset = C.PRESETS.get(preset_key)
        if preset is None:
            return  # unknown preset key — leave fields untouched
        use_cuda = self.has_cuda

        # "auto" compute type lets CTranslate2 pick float16 on CUDA and int8 on
        # CPU, which are the fastest accurate choices on each.
        self.combo_device.set("cuda" if use_cuda else "cpu")
        self.combo_compute.set("auto")

        self.set_mode_selection("batched" if preset["batched"] else "sequential")
        self.set_entry_value(
            self.entry_batch,
            preset["batch_size_cuda"] if use_cuda else preset["batch_size_cpu"],
        )
        self.set_entry_value(self.entry_beam_size, preset["beam_size"])
        self.set_entry_value(self.entry_vad_min_speech, preset["vad_min_speech_ms"])
        self.set_entry_value(self.entry_vad_silence, preset["vad_min_silence_ms"])
        self.set_entry_value(self.entry_vad_pad, preset["vad_speech_pad_ms"])
        self.set_entry_value(self.entry_no_speech, preset["no_speech_threshold"])
        self.set_entry_value(self.entry_compression_ratio, preset["compression_ratio_threshold"])
        self.set_entry_value(self.entry_log_prob, preset["log_prob_threshold"])
        # Blank means "off" — the field is optional, unlike the others.
        self.set_entry_value(self.entry_hallucination,
                             preset["hallucination_silence_threshold"] or "")

        self._set_check(self.check_temp_fallback, preset["temperature_fallback"])
        self._set_check(self.check_condition_prev, preset["condition_on_previous_text"])
        self._set_check(self.check_word_timestamps, preset["word_timestamps"])
        self._set_check(self.check_vad, preset["vad_enabled"])

        self.slider_vad.set(preset["vad_threshold"])
        self.update_vad_label(preset["vad_threshold"])

        self.refresh_runtime_summary()

    @staticmethod
    def _set_check(checkbox, enabled: bool) -> None:
        checkbox.select() if enabled else checkbox.deselect()

    def selected_formats(self) -> list[str]:
        """Output formats currently ticked, in the canonical constants order."""
        if not hasattr(self, "format_checkboxes"):
            return list(C.DEFAULT_OUTPUT_FORMATS)
        return [fmt for fmt in C.OUTPUT_FORMATS if self.format_checkboxes[fmt].get()]

    def update_vad_label(self, val):
        self.lbl_vad_val.configure(text=f"{float(val):.2f}")

    # ------------------------------------------------------------------
    # Phase F: hotkeys, drag&drop, theme, post-run actions, input lock
    # ------------------------------------------------------------------
    def _bind_hotkeys(self) -> None:
        """Wire up F5 / Esc / Ctrl+O / Ctrl+L / Ctrl+Q at the root level.

        We use bind_all so hotkeys still fire when focus is inside an Entry
        or the log textbox. Each binding short-circuits with a no-op if the
        action is invalid for the current state (e.g. F5 while a job runs).
        """
        self.bind_all("<F5>", lambda _e: self._hotkey_start())
        self.bind_all("<Escape>", lambda _e: self._hotkey_stop())
        self.bind_all("<Control-o>", lambda _e: self._hotkey_browse_audio())
        self.bind_all("<Control-O>", lambda _e: self._hotkey_browse_audio())
        self.bind_all("<Control-l>", lambda _e: self._hotkey_focus_logs())
        self.bind_all("<Control-L>", lambda _e: self._hotkey_focus_logs())
        self.bind_all("<Control-q>", lambda _e: self.on_close())
        self.bind_all("<Control-Q>", lambda _e: self.on_close())

    def _hotkey_start(self) -> None:
        if not self.is_running:
            self.start_process()

    def _hotkey_stop(self) -> None:
        if self.is_running:
            self.stop_process()

    def _hotkey_browse_audio(self) -> None:
        if not self.is_running:
            self.browse_audio()

    def _hotkey_focus_logs(self) -> None:
        try:
            self.tabview.set(self.tab_names["logs"])
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _wire_drag_and_drop(self) -> None:
        """Hook windnd onto the audio entry if the optional dep is present.

        Drop semantics: a directory becomes the new audio folder; a file's
        parent directory becomes the new audio folder. We never copy files —
        the worker scans the folder anyway.
        """
        if not _HAS_WINDND or windnd is None or not hasattr(self, "entry_audio"):
            return
        try:
            windnd.hook_dropfiles(self.entry_audio, func=self._on_drop_files)
        except Exception:
            pass  # best-effort; D&D is a nice-to-have

    def _on_drop_files(self, files) -> None:
        if not files:
            return
        # windnd hands us bytes on Windows — decode with the active code page
        # then fall back to utf-8 with replacement.
        first = files[0]
        if isinstance(first, bytes):
            try:
                first = first.decode("mbcs")
            except Exception:
                first = first.decode("utf-8", errors="replace")
        path = Path(first)
        target = path if path.is_dir() else path.parent
        try:
            self.entry_audio.delete(0, "end")
            self.entry_audio.insert(0, str(target))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _collect_input_widgets(self) -> None:
        """Snapshot widgets that should grey out during a run.

        Keeping this as an explicit list avoids an introspection walk every
        time we toggle, and we deliberately exclude btn_start / btn_stop /
        progress / banner / theme switch — those have their own state.
        """
        names = [
            "entry_audio", "btn_browse_audio",
            "entry_output", "btn_browse_output",
            "combo_model", "btn_refresh_models", "btn_model_folder",
            "combo_lang", "check_auto_lang",
            "combo_preset", "combo_ui_lang",
            "entry_initial_prompt",
            "combo_device", "combo_compute", "combo_mode", "entry_batch",
            "check_vad", "slider_vad", "entry_vad_min_speech", "entry_vad_silence",
            "entry_vad_pad",
            "entry_beam_size", "check_temp_fallback", "check_condition_prev",
            "check_word_timestamps", "entry_no_speech", "entry_compression_ratio",
            "entry_log_prob", "entry_hallucination",
        ]
        widgets = [getattr(self, n) for n in names if hasattr(self, n)]
        widgets.extend(getattr(self, "format_checkboxes", {}).values())
        self._input_widgets = widgets

    def _set_inputs_state(self, enabled: bool) -> None:
        """Enable/disable every collected input. Defensive: per-widget try."""
        state = "normal" if enabled else "disabled"
        for widget in self._input_widgets:
            try:
                widget.configure(state=state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _set_post_run_actions(self, enabled: bool) -> None:
        """Light up the 'Open folder' and 'View result' buttons after a run."""
        state = "normal" if enabled else "disabled"
        if hasattr(self, "btn_open_folder"):
            self.btn_open_folder.configure(state=state)
        if hasattr(self, "btn_view_result"):
            # 'View result' only makes sense when we actually have a file.
            view_state = state if (enabled and self.last_result_path) else "disabled"
            self.btn_view_result.configure(state=view_state)

    def on_core_result(self, audio_path: Path, written: list[Path]) -> None:
        """Called from the worker thread with the files the core actually wrote."""
        self.result_queue.put((audio_path, list(written)))

    def _track_finished_file(self, written: list[Path]) -> None:
        """Remember what to open for 'View result' / 'Open folder'.

        Driven by the real paths the core reports rather than by guessing
        `<stem>.txt`, which is wrong as soon as the user picks, say, Markdown
        and JSON but not plain text.
        """
        existing = [p for p in written if p.exists()]
        if not existing:
            return
        self.last_output_dir = existing[0].parent
        preferred = pick_preview(existing)
        if preferred is not None:
            self.last_result_path = preferred

    def open_results_folder(self) -> None:
        """Open the configured output folder in the OS file manager."""
        target = self.last_output_dir or Path(self.entry_output.get().strip() or ".")
        try:
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as e:
            self.error_banner.show(
                self.t("banner_open_folder_failed", error=e), level="error",
            )

    def preview_latest_result(self) -> None:
        """Spawn a PreviewDialog for the most recently finished transcript."""
        if not self.last_result_path or not self.last_result_path.exists():
            self.error_banner.show(self.t("banner_no_results_yet"), level="warn")
            return
        open_preview(self, self.last_result_path, self.t)

    # ------------------------------------------------------------------
    # Per-file progress + ETA (Phase E)
    # ------------------------------------------------------------------
    def _reset_file_progress(self) -> None:
        """Clear the tracker; called at the top of every start_process."""
        self.files_total = 0
        self.current_file_index = 0
        self.current_file_name = ""
        self._file_started_at = 0.0
        self._completed_durations = []

    def _note_file_started(self, name: str) -> None:
        """Mark `name` as the currently-processing file and start its timer."""
        self.current_file_index += 1
        self.current_file_name = name
        self._file_started_at = time.monotonic()

    def _note_file_done(self) -> None:
        """Record elapsed time for the just-finished file (used for ETA)."""
        if self._file_started_at:
            elapsed = time.monotonic() - self._file_started_at
            if elapsed > 0:
                self._completed_durations.append(elapsed)
        self._file_started_at = 0.0

    def _estimate_eta_seconds(self) -> float:
        """Median of completed durations × remaining files. 0 if unknown."""
        if not self._completed_durations or self.files_total == 0:
            return 0.0
        remaining = max(0, self.files_total - self.current_file_index)
        if remaining == 0:
            return 0.0
        per_file = statistics.median(self._completed_durations)
        return per_file * remaining

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format `seconds` as 'm:ss' (or 'h:mm:ss' for runs over an hour)."""
        if seconds <= 0:
            return ""
        total = int(round(seconds))
        if total >= 3600:
            h, rem = divmod(total, 3600)
            m, s = divmod(rem, 60)
            return f"{h}:{m:02d}:{s:02d}"
        m, s = divmod(total, 60)
        return f"{m}:{s:02d}"

    def _format_status_for_display(self, status: str) -> str:
        """Translate a raw core status to a localized display string.

        Side-effect: updates per-file tracking when the status indicates a
        file boundary so the resulting text can include 'File N/M: name'
        plus an ETA estimate.
        """
        # File-boundary statuses are parsed rather than string-matched here; the
        # wire format has a single owner in whisper_engine.progress.
        parsed = parse_status(status)
        if parsed.kind == "processing":
            self._note_file_started(parsed.name)
        elif parsed.kind == "done_file":
            self._note_file_done()

        # Translate the base status using the existing translator.
        base = self.translate_status(status)

        # Only decorate with file counter while a job is in flight.
        if self.is_running and self.files_total > 0 and self.current_file_index > 0:
            file_line = self.t(
                "progress_file_line",
                idx=self.current_file_index,
                total=self.files_total,
                name=self.current_file_name or "",
            )
            eta_secs = self._estimate_eta_seconds()
            if eta_secs > 0:
                eta_text = self.t("progress_eta", eta=self._format_eta(eta_secs))
                return f"{file_line}\n{base} — {eta_text}"
            return f"{file_line}\n{base}"

        return base

    # ------------------------------------------------------------------
    # Pre-flight validation (Phase D)
    # ------------------------------------------------------------------
    def _preflight_validate(self) -> bool:
        """Return True iff every ValidatedEntry currently holds a valid value.

        On failure: logs each offending field, pops a localized messagebox
        listing them, and returns False so start_process can abort early.
        Each entry's red border was already applied by ValidatedEntry on
        the input event; this method only handles aggregation + reporting.
        """
        problems = []
        for entry in self.validated_entries:
            if entry.is_valid():
                continue
            err = entry.get_error()
            if err is None:
                continue  # defensive — shouldn't happen
            key, kwargs = err
            label_text = self.t(entry.label_key) if entry.label_key else ""
            # label_text already ends with a colon (e.g. "Batch size:"), so
            # we just append a space and the translated reason.
            problems.append(f"{label_text} {self.t(key, **kwargs)}".strip())

        if not problems:
            return True

        title = self.t("validation_failed_title")
        intro = self.t("banner_validation_intro")
        # Log + banner so the error is visible no matter which tab is open.
        self.log(title)
        for line in problems:
            self.log(f"  - {line}")
        self.error_banner.show(intro + "\n  " + "\n  ".join(problems), level="error")
        return False

    # ------------------------------------------------------------------
    # Settings persistence (Phase B)
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_int(value, default):
        try:
            return int(float(str(value).strip()))
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(value, default):
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _optional_float(value):
        """Parse an optional numeric field. Blank means "off", i.e. None."""
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    def _apply_persisted_entry(self, entry, value):
        """Write `value` into a CTkEntry only if it looks meaningful.

        Empty strings and None are skipped so the preset-driven default that
        was already inserted in apply_preset() stays put.
        """
        if value is None or value == "":
            return
        self.set_entry_value(entry, value)

    def _load_persisted_settings(self):
        """Apply ~/.whisper_gui/settings.json on top of preset-fast defaults.

        This runs after build_*_tab and apply_preset("fast"), so individual
        keys override the preset baseline and missing keys keep the baseline.
        Never raises: on any malformed value we silently fall through.
        """
        self.settings = settings_store.load()
        s = self.settings

        # Window geometry — best-effort, Tk silently rejects garbage strings.
        geometry = s.get("window_geometry")
        if isinstance(geometry, str) and geometry:
            try:
                self.geometry(geometry)
            except Exception:
                pass

        # UI language: switch + relocalize labels if different.
        lang = s.get("ui_language")
        if lang in C.UI_LANGUAGES and lang != self.ui_language:
            self.ui_language = lang
            self.apply_localization()

        # Preset (re-apply so its baseline matches what the user last chose;
        # later we override individual fields the user tweaked manually).
        # Always reflect the saved preset in the combo, even when it matches the
        # default — otherwise the sidebar shows one preset while the settings
        # file holds another.
        preset_key = s.get("preset")
        if preset_key in self.preset_keys:
            self.set_preset_selection(preset_key)
            if preset_key != C.DEFAULT_PRESET:
                self.apply_preset(preset_key)

        # Files
        audio_dir = s.get("audio_dir")
        if isinstance(audio_dir, str) and audio_dir:
            self.entry_audio.delete(0, "end")
            self.entry_audio.insert(0, audio_dir)
        output_dir = s.get("output_dir")
        if isinstance(output_dir, str) and output_dir:
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, output_dir)

        # Model — make sure custom paths still appear in the combo's list.
        model_id = s.get("model_id")
        if isinstance(model_id, str) and model_id:
            current_values = list(self.combo_model.cget("values") or [])
            # A model that no longer exists on disk must not be silently selected;
            # only offer it if it is a converted directory or an explicit path.
            if model_id in current_values or Path(model_id).is_dir():
                if model_id not in current_values:
                    current_values.append(model_id)
                    self.combo_model.configure(values=current_values)
                self.combo_model.set(model_id)

        # Transcription language + auto-detect
        trans_lang = s.get("transcription_language")
        if trans_lang in C.TRANSCRIPTION_LANGUAGES:
            self.combo_lang.set(trans_lang)
        self._set_check(self.check_auto_lang, bool(s.get("auto_lang")))
        self._apply_persisted_entry(self.entry_initial_prompt, s.get("initial_prompt"))

        # Compute (clamp cuda→cpu when no GPU is present)
        device = s.get("device")
        if device in C.DEVICES:
            if device == "cuda" and not self.has_cuda:
                device = "cpu"
            self.combo_device.set(device)
        compute_type = s.get("compute_type")
        if compute_type in C.COMPUTE_TYPES:
            self.combo_compute.set(compute_type)
        if "batched" in s:
            self.set_mode_selection("batched" if s.get("batched") else "sequential")

        # Numeric overrides on top of the preset baseline
        self._apply_persisted_entry(self.entry_batch, s.get("batch_size"))
        self._apply_persisted_entry(self.entry_beam_size, s.get("beam_size"))
        self._apply_persisted_entry(self.entry_vad_min_speech, s.get("vad_min_speech_ms"))
        self._apply_persisted_entry(self.entry_vad_silence, s.get("vad_min_silence_ms"))
        self._apply_persisted_entry(self.entry_vad_pad, s.get("vad_speech_pad_ms"))
        self._apply_persisted_entry(self.entry_no_speech, s.get("no_speech_threshold"))
        self._apply_persisted_entry(self.entry_compression_ratio,
                                    s.get("compression_ratio_threshold"))
        self._apply_persisted_entry(self.entry_log_prob, s.get("log_prob_threshold"))
        # Optional field: an explicit null means "off", so clear rather than skip.
        if "hallucination_silence_threshold" in s:
            self.set_entry_value(self.entry_hallucination,
                                 s.get("hallucination_silence_threshold") or "")

        for key, checkbox in (
            ("temperature_fallback", self.check_temp_fallback),
            ("condition_on_previous_text", self.check_condition_prev),
            ("word_timestamps", self.check_word_timestamps),
            ("vad_filter", self.check_vad),
        ):
            if key in s:
                self._set_check(checkbox, bool(s.get(key)))

        vad_threshold = s.get("vad_threshold")
        if isinstance(vad_threshold, int | float):
            self.slider_vad.set(float(vad_threshold))
            self.update_vad_label(float(vad_threshold))

        formats = s.get("formats")
        if isinstance(formats, list) and any(f in C.OUTPUT_FORMATS for f in formats):
            for fmt, checkbox in self.format_checkboxes.items():
                self._set_check(checkbox, fmt in formats)

        # Reflect any device/model/lang change in the runtime summary text.
        self.refresh_runtime_summary()

    def _collect_settings_snapshot(self) -> dict:
        """Read current widget values into a JSON-serializable settings dict."""
        defaults = settings_store.DEFAULTS
        try:
            geometry = self.winfo_geometry()
        except Exception:
            geometry = C.WINDOW_GEOMETRY
        return {
            "schema_version": settings_store.SCHEMA_VERSION,
            "window_geometry": geometry,
            "ui_language": self.ui_language,
            "transcription_language": self.combo_lang.get() if hasattr(self, "combo_lang") else defaults["transcription_language"],
            "auto_lang": bool(self.check_auto_lang.get()) if hasattr(self, "check_auto_lang") else False,
            "preset": self.selected_preset_key(),
            "audio_dir": self.entry_audio.get().strip() if hasattr(self, "entry_audio") else "",
            "output_dir": self.entry_output.get().strip() if hasattr(self, "entry_output") else "",
            "initial_prompt": self.entry_initial_prompt.get().strip() if hasattr(self, "entry_initial_prompt") else "",
            "model_id": self.combo_model.get().strip() if hasattr(self, "combo_model") else defaults["model_id"],
            "device": self.combo_device.get() if hasattr(self, "combo_device") else defaults["device"],
            "compute_type": self.combo_compute.get() if hasattr(self, "combo_compute") else defaults["compute_type"],
            "batched": self.selected_mode_key() == "batched",
            "batch_size": self._safe_int(self.entry_batch.get(), defaults["batch_size"]) if hasattr(self, "entry_batch") else defaults["batch_size"],
            "beam_size": self._safe_int(self.entry_beam_size.get(), defaults["beam_size"]) if hasattr(self, "entry_beam_size") else defaults["beam_size"],
            "temperature_fallback": bool(self.check_temp_fallback.get()) if hasattr(self, "check_temp_fallback") else defaults["temperature_fallback"],
            "condition_on_previous_text": bool(self.check_condition_prev.get()) if hasattr(self, "check_condition_prev") else defaults["condition_on_previous_text"],
            "word_timestamps": bool(self.check_word_timestamps.get()) if hasattr(self, "check_word_timestamps") else defaults["word_timestamps"],
            "no_speech_threshold": self._safe_float(self.entry_no_speech.get(), defaults["no_speech_threshold"]) if hasattr(self, "entry_no_speech") else defaults["no_speech_threshold"],
            "compression_ratio_threshold": self._safe_float(self.entry_compression_ratio.get(), defaults["compression_ratio_threshold"]) if hasattr(self, "entry_compression_ratio") else defaults["compression_ratio_threshold"],
            "log_prob_threshold": self._safe_float(self.entry_log_prob.get(), defaults["log_prob_threshold"]) if hasattr(self, "entry_log_prob") else defaults["log_prob_threshold"],
            "hallucination_silence_threshold": self._optional_float(self.entry_hallucination.get()) if hasattr(self, "entry_hallucination") else None,
            "vad_filter": bool(self.check_vad.get()) if hasattr(self, "check_vad") else defaults["vad_filter"],
            "vad_threshold": float(self.slider_vad.get()) if hasattr(self, "slider_vad") else defaults["vad_threshold"],
            "vad_min_speech_ms": self._safe_int(self.entry_vad_min_speech.get(), defaults["vad_min_speech_ms"]) if hasattr(self, "entry_vad_min_speech") else defaults["vad_min_speech_ms"],
            "vad_min_silence_ms": self._safe_int(self.entry_vad_silence.get(), defaults["vad_min_silence_ms"]) if hasattr(self, "entry_vad_silence") else defaults["vad_min_silence_ms"],
            "vad_speech_pad_ms": self._safe_int(self.entry_vad_pad.get(), defaults["vad_speech_pad_ms"]) if hasattr(self, "entry_vad_pad") else defaults["vad_speech_pad_ms"],
            "formats": self.selected_formats() or list(C.DEFAULT_OUTPUT_FORMATS),
        }

    def on_close(self):
        """Persist current GUI state then destroy the window.

        Save errors are non-fatal: the user closing the app must always be able
        to close the app, even if the settings file is unwritable.
        """
        try:
            settings_store.save(self._collect_settings_snapshot())
        except Exception:
            pass
        try:
            if self.is_running:
                self.core.request_stop()
        except Exception:
            pass
        self.destroy()

    def browse_audio(self):
        directory = filedialog.askdirectory()
        if directory:
            self.entry_audio.delete(0, "end")
            self.entry_audio.insert(0, directory)

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, directory)

    def log(self, msg: str):
        self.log_queue.put(msg)

    def on_core_log(self, msg: str):
        self.log_queue.put(msg)

    def on_core_progress(self, percent: float, status: str):
        if status == "Done.":
            status = "Done"
        self.progress_queue.put((percent, status))

    def process_queues(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.txt_log.configure(state="normal")
            self.txt_log.insert("end", msg + "\n")
            self.txt_log.see("end")
            self.txt_log.configure(state="disabled")

        # Drained before the progress queue: a terminal status enables the
        # post-run buttons, which need last_result_path to already be set.
        while not self.result_queue.empty():
            _audio_path, written = self.result_queue.get()
            self._track_finished_file(written)

        while not self.progress_queue.empty():
            percent, status = self.progress_queue.get()
            self.current_status_raw = status
            self.progress_bar.set(percent)
            self.lbl_percentage.configure(text=f"{int(percent * 100)}%")
            # Per-file tracker watches every status; the returned text already
            # includes "File N/M: name" + ETA when those make sense, so the
            # status label can just show it directly.
            display_text = self._format_status_for_display(status)
            self.status_label.configure(text=display_text)

            if status == "Done" or "Stopped" in status or "Error" in status:
                self.btn_start.configure(state="normal")
                self.btn_stop.configure(state="disabled")
                self.is_running = False
                # Phase F: re-enable inputs and (if a transcript landed)
                # light up the post-run actions.
                self._set_inputs_state(True)
                self._set_post_run_actions(True)
                if status == "Done":
                    self.lbl_percentage.configure(text="100%")

        # Banner messages from background threads. Drained on Tk thread.
        while not self.banner_queue.empty():
            level, message = self.banner_queue.get()
            self.error_banner.show(message, level=level)

        self.after(100, self.process_queues)

    def start_process(self):
        if self.is_running:
            return

        # Every new run starts with a clean banner and a zeroed per-file
        # tracker; the old tracker would show stale ETA.
        self.error_banner.hide()
        self._reset_file_progress()

        # Pre-flight: every ValidatedEntry registers itself in
        # self.validated_entries. If any has a current error we refuse to
        # start, log the offending fields, and surface them on the banner.
        if not self._preflight_validate():
            return

        if self.combo_device.get() == "cpu":
            response = messagebox.askokcancel(
                self.t("cpu_warning_title"),
                self.t("cpu_warning_body"),
            )
            if not response:
                return

        # The whole config-build + file-scan block is wrapped so any
        # unexpected exception (bad output_dir path characters, read-only
        # folder, etc.) lands on the banner instead of dying silently.
        formats = self.selected_formats()
        if not formats:
            self.error_banner.show(self.t("banner_no_formats"), level="error")
            return

        try:
            cfg = TranscriptionConfig(
                model_id=self.combo_model.get().strip(),
                lang=self.combo_lang.get(),
                auto_lang=bool(self.check_auto_lang.get()),
                initial_prompt=self.entry_initial_prompt.get().strip() or None,
                device=self.combo_device.get(),
                compute_type=self.combo_compute.get(),
                batched=self.selected_mode_key() == "batched",
                batch_size=int(self.entry_batch.get()),
                beam_size=int(self.entry_beam_size.get()),
                temperature_fallback=bool(self.check_temp_fallback.get()),
                condition_on_previous_text=bool(self.check_condition_prev.get()),
                word_timestamps=bool(self.check_word_timestamps.get()),
                no_speech_threshold=float(self.entry_no_speech.get()),
                compression_ratio_threshold=float(self.entry_compression_ratio.get()),
                log_prob_threshold=float(self.entry_log_prob.get()),
                hallucination_silence_threshold=self._optional_float(
                    self.entry_hallucination.get()),
                vad_filter=bool(self.check_vad.get()),
                vad_threshold=float(self.slider_vad.get()),
                vad_min_speech_ms=int(self.entry_vad_min_speech.get()),
                vad_min_silence_ms=int(self.entry_vad_silence.get()),
                vad_speech_pad_ms=int(self.entry_vad_pad.get()),
                formats=tuple(formats),
                output_dir=Path(self.entry_output.get().strip()),
            )
        except (ValueError, OSError) as e:
            self.log(self.t("log_config_error", error=e))
            self.error_banner.show(self.t("banner_start_failed", error=e), level="error")
            return

        # Surface anything the config had to override, so the Advanced tab never
        # claims a setting that faster-whisper will discard.
        if cfg.notes:
            self.error_banner.show("\n".join(cfg.notes), level="warn")

        audio_dir = Path(self.entry_audio.get().strip())
        if not audio_dir.exists():
            msg = self.t("log_audio_folder_not_found", path=audio_dir)
            self.log(msg)
            self.error_banner.show(
                self.t("banner_audio_dir_missing", path=audio_dir), level="error",
            )
            return

        self.files_to_process = find_audio_files(audio_dir)

        if not self.files_to_process:
            self.log(self.t("log_no_audio_files"))
            self.error_banner.show(
                self.t("banner_no_audio_files", path=audio_dir), level="warn",
            )
            return

        self.log(self.t("log_found_files", count=len(self.files_to_process)))

        # Arm per-file progress tracking before the worker thread starts
        # so the first "Processing ..." status already knows the total.
        self.files_total = len(self.files_to_process)
        self.is_running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress_bar.set(0)
        # Phase F: lock inputs and clear stale post-run state so the
        # "View result" button doesn't open the previous run's file.
        self._set_inputs_state(False)
        self.last_result_path = None
        self._set_post_run_actions(False)

        worker = threading.Thread(target=self.run_thread, args=(cfg,))
        worker.daemon = True
        worker.start()

    def run_thread(self, cfg):
        try:
            self.core.process_files(self.files_to_process, cfg)
        except Exception as e:
            # Worker thread cannot touch Tk widgets directly; route the
            # banner update through the Tk-thread-drained banner_queue.
            self.on_core_log(self.t("log_critical_worker_error", error=e))
            self.banner_queue.put(
                ("error", self.t("banner_worker_crashed", error=e))
            )
            self.on_core_progress(0, "Error")

    def stop_process(self):
        if self.is_running:
            self.core.request_stop()
            self.log(self.t("log_stopping"))
            self.btn_stop.configure(state="disabled")


if __name__ == "__main__":
    app = WhisperGUI()
    app.mainloop()
