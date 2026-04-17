# -*- coding: utf-8 -*-
from tkinter import filedialog, messagebox
import customtkinter as ctk
import threading
import queue
import sys
import torch
from pathlib import Path
from typing import List

try:
    from whisper_core import WhisperCore, TranscriptionConfig
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from whisper_core import WhisperCore, TranscriptionConfig

# Single source of truth for theme/colors/sizes/lists/presets lives in gui/.
from gui import constants as C
from gui import settings_store
from gui.i18n import UI_TEXT
from gui.widgets import settings_tab as settings_tab_builder
from gui.widgets import advanced_tab as advanced_tab_builder
from gui.widgets import logs_tab as logs_tab_builder


ctk.set_appearance_mode(C.APPEARANCE_MODE)
ctk.set_default_color_theme(C.COLOR_THEME)


class WhisperGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(C.WINDOW_TITLE)
        self.geometry(C.WINDOW_GEOMETRY)
        self.minsize(*C.WINDOW_MIN_SIZE)
        self.configure(fg_color=C.COLOR_BG)

        self.core = WhisperCore(on_log=self.on_core_log, on_progress=self.on_core_progress)
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.is_running = False
        self.files_to_process: List[Path] = []
        self.ui_language = C.DEFAULT_UI_LANGUAGE
        self.preset_keys = list(C.PRESET_KEYS)
        self.decode_profile_keys = list(C.DECODE_PROFILE_KEYS)
        self.localized_help_labels = []
        self.current_status_raw = "Ready"
        self.has_cuda = torch.cuda.is_available()
        self.has_local_vad = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_frame = ctk.CTkFrame(
            self, width=C.SIDEBAR_WIDTH, corner_radius=0, fg_color=C.COLOR_SIDEBAR_BG,
        )
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
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

        self.sidebar_hint_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=self.t("sidebar_hint"),
            text_color=C.COLOR_TEXT_HINT,
            font=ctk.CTkFont(size=C.HELP_FONT_SIZE),
            wraplength=C.WRAPLENGTH_SIDEBAR,
            justify="left",
        )
        self.sidebar_hint_label.grid(row=10, column=0, padx=20, pady=(0, 18), sticky="w")

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
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

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
        self.set_preset_selection("fast")
        self.apply_preset("fast")

        # Load persisted user settings (window geometry, last preset, paths,
        # tweaked numeric fields). This must run AFTER apply_preset("fast") so
        # individual saved field values can override the preset baseline.
        self._load_persisted_settings()

        # Persist current values when the user closes the window.
        self.protocol("WM_DELETE_WINDOW", self.on_close)

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

    def localized_preset_values(self) -> List[str]:
        return [self.preset_display_text(key) for key in self.preset_keys]

    def preset_key_from_value(self, value: str) -> str:
        normalized = value.strip()
        for language_code in UI_TEXT:
            for preset_key in self.preset_keys:
                if normalized == UI_TEXT[language_code][f"preset_{preset_key}"]:
                    return preset_key
        return "fast"

    def selected_preset_key(self) -> str:
        return self.preset_key_from_value(self.combo_preset.get()) if hasattr(self, "combo_preset") else "fast"

    def set_preset_selection(self, preset_key: str):
        if hasattr(self, "combo_preset"):
            self.combo_preset.set(self.preset_display_text(preset_key))

    def decode_profile_label(self, profile_key: str) -> str:
        return self.t(f"decode_{profile_key}")

    def localized_decode_profile_values(self) -> List[str]:
        return [self.decode_profile_label(key) for key in self.decode_profile_keys]

    def decode_profile_key_from_value(self, value: str) -> str:
        normalized = value.strip()
        for language_code in UI_TEXT:
            for profile_key in self.decode_profile_keys:
                if normalized == UI_TEXT[language_code][f"decode_{profile_key}"]:
                    return profile_key
        return "balanced"

    def selected_decode_profile_key(self) -> str:
        return self.decode_profile_key_from_value(self.combo_decode.get()) if hasattr(self, "combo_decode") else "balanced"

    def set_decode_profile_selection(self, profile_key: str):
        if hasattr(self, "combo_decode"):
            self.combo_decode.set(self.decode_profile_label(profile_key))

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

    def refresh_vad_status_text(self):
        if not hasattr(self, "check_vad"):
            return

        if not self.has_local_vad:
            preferred_vad_path = self.core.vad_repo_hints()[0]
            self.check_vad.configure(text=self.t("checkbox_enable_vad_missing"))
            self.vad_hint_label.configure(text=self.t("vad_hint_path", path=preferred_vad_path))
        else:
            self.check_vad.configure(text=self.t("checkbox_enable_vad"))
            self.vad_hint_label.configure(text=self.t("vad_hint_repo", path=self.core.resolve_local_vad_repo()))

    def apply_localization(self):
        current_tab = self.current_tab_key()
        current_preset = self.selected_preset_key()
        current_decode = self.selected_decode_profile_key() if hasattr(self, "combo_decode") else "balanced"

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
        self.output_section_label.configure(text=self.t("section_output"))
        self.check_srt.configure(text=self.t("checkbox_srt"))

        self.adv_intro_title_label.configure(text=self.t("adv_intro_title"))
        self.adv_intro_body_label.configure(text=self.t("adv_intro_body"))
        self.compute_section_label.configure(text=self.t("section_compute"))
        self.device_label.configure(text=self.t("label_device"))
        self.precision_label.configure(text=self.t("label_precision"))
        self.batch_size_label.configure(text=self.t("label_batch_size"))
        self.use_vad_label.configure(text=self.t("label_use_vad"))
        self.segmentation_section_label.configure(text=self.t("section_segmentation"))
        self.vad_threshold_label.configure(text=self.t("label_vad_threshold"))
        self.decode_profile_label_widget.configure(text=self.t("label_decode_profile"))
        self.combo_decode.configure(values=self.localized_decode_profile_values())
        self.set_decode_profile_selection(current_decode)
        self.target_db_label.configure(text=self.t("label_target_db"))
        self.merge_gap_label.configure(text=self.t("label_merge_gap"))
        self.min_silence_label.configure(text=self.t("label_min_silence"))
        self.chunk_sec_label.configure(text=self.t("label_chunk_sec"))
        self.overlap_sec_label.configure(text=self.t("label_overlap_sec"))
        self.max_new_tokens_label.configure(text=self.t("label_max_new_tokens"))

        self.logs_intro_title_label.configure(text=self.t("logs_intro_title"))
        self.logs_intro_body_label.configure(text=self.t("logs_intro_body"))

        for label, key in self.localized_help_labels:
            label.configure(text=self.t(key))

        self.refresh_vad_status_text()
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
        if clean_status.startswith("Processing "):
            return f"{self.t('status_processing_file_prefix')}{clean_status[len('Processing '):]}"
        if clean_status.startswith("Pipeline: "):
            return f"{self.t('status_processing_file_prefix')}{clean_status[len('Pipeline: '):]}"
        if clean_status.startswith("transcribing:"):
            # Neutral status code emitted by whisper_core. Format: "transcribing:<pct>"
            return self.t("status_processing_percent", percent=clean_status.split(":", 1)[1].strip())
        # Legacy: older cores emitted the Russian hardcoded string directly.
        if clean_status.startswith("Обработка:"):
            return self.t("status_processing_percent", percent=clean_status.split(":", 1)[1].strip())
        return clean_status

    def check_hardware(self):
        self.has_cuda = torch.cuda.is_available()
        self.has_local_vad = self.core.resolve_local_vad_repo() is not None
        if not self.has_cuda:
            messagebox.showwarning(
                self.t("hardware_check_title"),
                self.t("hardware_check_body"),
            )
        if not self.has_local_vad:
            self.check_vad.deselect()
        self.refresh_vad_status_text()
        self.refresh_runtime_summary()

    def short_model_name(self, model_value: str) -> str:
        if Path(model_value).exists():
            return Path(model_value).name
        return model_value.split("/")[-1]

    def describe_model(self, model_value: str) -> str:
        lower = model_value.lower()
        if Path(model_value).exists():
            return self.t("model_desc_custom")
        if "medium" in lower:
            return self.t("model_desc_medium")
        if "small" in lower:
            return self.t("model_desc_small")
        if "large" in lower:
            return self.t("model_desc_large")
        return self.t("model_desc_local_generic")

    def refresh_runtime_summary(self):
        preset = self.selected_preset_key()
        model_value = self.combo_model.get().strip() if hasattr(self, "combo_model") else "openai/whisper-medium"

        self.preset_summary_label.configure(text=self.preset_description(preset))

        compute_value = self.combo_device.get() if hasattr(self, "combo_device") else ("cuda" if self.has_cuda else "cpu")
        runtime_lines = [
            self.t("runtime_model_line", model=self.short_model_name(model_value)),
            self.t("runtime_compute_line", compute=compute_value.upper()),
            self.t("runtime_mode_line"),
            self.t("runtime_vad_found_line" if self.has_local_vad else "runtime_vad_missing_line"),
        ]
        self.runtime_summary_label.configure(text="\n".join(runtime_lines))

        if hasattr(self, "model_help_label"):
            self.model_help_label.configure(text=self.describe_model(model_value))

    def set_entry_value(self, entry: ctk.CTkEntry, value):
        entry.delete(0, "end")
        entry.insert(0, str(value))

    def discover_local_models(self) -> List[str]:
        models_root = Path(__file__).parent / C.LOCAL_MODELS_SUBDIR
        if not models_root.exists():
            return []

        choices = []
        for cache_dir in sorted(models_root.glob("models--*")):
            snapshots_dir = cache_dir / "snapshots"
            if snapshots_dir.exists() and any(path.is_dir() for path in snapshots_dir.iterdir()):
                choices.append(cache_dir.name[len("models--"):].replace("--", "/"))
        return choices

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

        # Device + precision: presets prefer CUDA when available; "auto" dtype
        # lets WhisperCore pick bf16 on Ampere+ (cc>=8), fp16 on older CUDA,
        # and fp32 on CPU — bf16 is more numerically stable for Whisper.
        self.combo_device.set("cuda" if use_cuda else "cpu")
        self.combo_dtype.set("auto" if use_cuda else "float32")

        # Numeric fields driven entirely by the preset table.
        batch = preset["batch_size_cuda"] if use_cuda else preset["batch_size_cpu"]
        self.set_entry_value(self.entry_batch, batch)
        self.set_decode_profile_selection(preset["decode_profile"])
        self.set_entry_value(self.entry_target_db, preset["target_db"])
        self.set_entry_value(self.entry_vad_merge_gap, preset["vad_merge_gap"])
        self.set_entry_value(self.entry_vad_silence, preset["vad_silence_ms"])
        self.set_entry_value(self.entry_chunk_sec, preset["chunk_sec"])
        self.set_entry_value(self.entry_overlap_sec, preset["overlap_sec"])
        self.set_entry_value(self.entry_max_tokens, preset["max_new_tokens"])

        # VAD checkbox respects local availability; threshold is preset-driven.
        if preset["vad_enabled"] and self.has_local_vad:
            self.check_vad.select()
        else:
            self.check_vad.deselect()
        self.slider_vad.set(preset["vad_threshold"])
        self.update_vad_label(preset["vad_threshold"])

        self.refresh_runtime_summary()

    def update_vad_label(self, val):
        self.lbl_vad_val.configure(text=f"{float(val):.2f}")

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
        preset_key = s.get("preset")
        if preset_key in self.preset_keys and preset_key != "fast":
            self.set_preset_selection(preset_key)
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
        model_path = s.get("model_path")
        if isinstance(model_path, str) and model_path:
            current_values = list(self.combo_model.cget("values") or [])
            if model_path not in current_values:
                current_values.append(model_path)
                self.combo_model.configure(values=current_values)
            self.combo_model.set(model_path)

        # Transcription language + auto-detect
        trans_lang = s.get("transcription_language")
        if trans_lang in C.TRANSCRIPTION_LANGUAGES:
            self.combo_lang.set(trans_lang)
        if s.get("auto_lang"):
            self.check_auto_lang.select()
        elif "auto_lang" in s:
            self.check_auto_lang.deselect()

        # Compute (clamp cuda→cpu when no GPU is present)
        device = s.get("device")
        if device in C.DEVICES:
            if device == "cuda" and not self.has_cuda:
                device = "cpu"
            self.combo_device.set(device)
        dtype = s.get("dtype")
        if dtype in C.DTYPES:
            self.combo_dtype.set(dtype)

        # Numeric overrides on top of the preset baseline
        self._apply_persisted_entry(self.entry_batch, s.get("batch_size"))
        self._apply_persisted_entry(self.entry_max_tokens, s.get("max_new_tokens"))
        self._apply_persisted_entry(self.entry_chunk_sec, s.get("chunk_sec"))
        self._apply_persisted_entry(self.entry_overlap_sec, s.get("overlap_sec"))
        self._apply_persisted_entry(self.entry_target_db, s.get("target_db"))
        self._apply_persisted_entry(self.entry_vad_silence, s.get("vad_silence_ms"))
        self._apply_persisted_entry(self.entry_vad_merge_gap, s.get("vad_merge_gap"))

        # Decode profile
        decode = s.get("decode_profile")
        if decode in self.decode_profile_keys:
            self.set_decode_profile_selection(decode)

        # VAD checkbox + threshold (respect local availability for the box)
        if "use_vad" in s:
            if s.get("use_vad") and self.has_local_vad:
                self.check_vad.select()
            else:
                self.check_vad.deselect()
        vad_threshold = s.get("vad_threshold")
        if isinstance(vad_threshold, (int, float)):
            self.slider_vad.set(float(vad_threshold))
            self.update_vad_label(float(vad_threshold))

        # SRT toggle
        if "save_srt" in s:
            if s.get("save_srt"):
                self.check_srt.select()
            else:
                self.check_srt.deselect()

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
            "model_path": self.combo_model.get().strip() if hasattr(self, "combo_model") else defaults["model_path"],
            "device": self.combo_device.get() if hasattr(self, "combo_device") else defaults["device"],
            "dtype": self.combo_dtype.get() if hasattr(self, "combo_dtype") else defaults["dtype"],
            "batch_size": self._safe_int(self.entry_batch.get(), defaults["batch_size"]) if hasattr(self, "entry_batch") else defaults["batch_size"],
            "decode_profile": self.selected_decode_profile_key() if hasattr(self, "combo_decode") else defaults["decode_profile"],
            "max_new_tokens": self._safe_int(self.entry_max_tokens.get(), defaults["max_new_tokens"]) if hasattr(self, "entry_max_tokens") else defaults["max_new_tokens"],
            "chunk_sec": self._safe_float(self.entry_chunk_sec.get(), defaults["chunk_sec"]) if hasattr(self, "entry_chunk_sec") else defaults["chunk_sec"],
            "overlap_sec": self._safe_float(self.entry_overlap_sec.get(), defaults["overlap_sec"]) if hasattr(self, "entry_overlap_sec") else defaults["overlap_sec"],
            "target_db": self._safe_float(self.entry_target_db.get(), defaults["target_db"]) if hasattr(self, "entry_target_db") else defaults["target_db"],
            "use_vad": bool(self.check_vad.get()) if hasattr(self, "check_vad") else defaults["use_vad"],
            "vad_threshold": float(self.slider_vad.get()) if hasattr(self, "slider_vad") else defaults["vad_threshold"],
            "vad_silence_ms": self._safe_int(self.entry_vad_silence.get(), defaults["vad_silence_ms"]) if hasattr(self, "entry_vad_silence") else defaults["vad_silence_ms"],
            "vad_merge_gap": self._safe_float(self.entry_vad_merge_gap.get(), defaults["vad_merge_gap"]) if hasattr(self, "entry_vad_merge_gap") else defaults["vad_merge_gap"],
            "save_srt": bool(self.check_srt.get()) if hasattr(self, "check_srt") else False,
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

        while not self.progress_queue.empty():
            percent, status = self.progress_queue.get()
            self.current_status_raw = status
            self.progress_bar.set(percent)
            self.lbl_percentage.configure(text=f"{int(percent * 100)}%")
            self.status_label.configure(text=self.translate_status(status))

            if status == "Done" or "Stopped" in status or "Error" in status:
                self.btn_start.configure(state="normal")
                self.btn_stop.configure(state="disabled")
                self.is_running = False
                if status == "Done":
                    self.lbl_percentage.configure(text="100%")

        self.after(100, self.process_queues)

    def start_process(self):
        if self.is_running:
            return

        if self.combo_device.get() == "cpu":
            response = messagebox.askokcancel(
                self.t("cpu_warning_title"),
                self.t("cpu_warning_body"),
            )
            if not response:
                return

        try:
            cfg = TranscriptionConfig(
                model_path=self.combo_model.get().strip(),
                lang=self.combo_lang.get(),
                auto_lang=bool(self.check_auto_lang.get()),
                chunk_sec=float(self.entry_chunk_sec.get()),
                overlap_sec=float(self.entry_overlap_sec.get()),
                device=self.combo_device.get(),
                dtype=self.combo_dtype.get(),
                batch_size=int(self.entry_batch.get()),
                max_new_tokens=int(self.entry_max_tokens.get()),
                decode_profile=self.selected_decode_profile_key(),
                target_db=float(self.entry_target_db.get()),
                use_vad=bool(self.check_vad.get()),
                vad_threshold=float(self.slider_vad.get()),
                vad_min_silence_ms=int(self.entry_vad_silence.get()),
                vad_merge_gap_sec=float(self.entry_vad_merge_gap.get()),
                save_srt=bool(self.check_srt.get()),
                output_dir=Path(self.entry_output.get().strip()),
            )
        except ValueError as e:
            self.log(self.t("log_config_error", error=e))
            return

        audio_dir = Path(self.entry_audio.get().strip())
        if not audio_dir.exists():
            self.log(self.t("log_audio_folder_not_found", path=audio_dir))
            return

        self.files_to_process = []
        for ext in C.AUDIO_EXTENSIONS:
            self.files_to_process.extend(audio_dir.glob(ext))

        if not self.files_to_process:
            self.log(self.t("log_no_audio_files"))
            return

        self.files_to_process = sorted(self.files_to_process)
        self.log(self.t("log_found_files", count=len(self.files_to_process)))

        self.is_running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress_bar.set(0)

        worker = threading.Thread(target=self.run_thread, args=(cfg,))
        worker.daemon = True
        worker.start()

    def run_thread(self, cfg):
        try:
            self.core.process_files(self.files_to_process, cfg)
        except Exception as e:
            self.on_core_log(self.t("log_critical_worker_error", error=e))
            self.on_core_progress(0, "Error")

    def stop_process(self):
        if self.is_running:
            self.core.request_stop()
            self.log(self.t("log_stopping"))
            self.btn_stop.configure(state="disabled")


if __name__ == "__main__":
    app = WhisperGUI()
    app.mainloop()
