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


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


UI_TEXT = {
    "en": {
        "label_interface_language": "Interface language",
        "label_preset": "Preset",
        "preset_fast": "Fast",
        "preset_accurate": "Accurate",
        "preset_noisy": "Noisy audio",
        "preset_fast_desc": "Best default. Uses safer, quicker settings for everyday offline transcription.",
        "preset_accurate_desc": "Uses longer context and stricter decoding when wording matters more than speed.",
        "preset_noisy_desc": "Keeps speech together more carefully for messy recordings with noise or broken pauses.",
        "subtitle": "Local-first transcription workspace",
        "button_start": "Start Transcription",
        "button_stop": "Stop",
        "sidebar_hint": "If you are unsure, keep medium selected, choose Fast, and start from the Settings tab.",
        "status_ready": "Ready",
        "status_done": "Done",
        "status_error": "Error",
        "status_stopped": "Stopped",
        "status_processing_file_prefix": "Processing ",
        "status_processing_percent": "Processing: {percent}",
        "tab_settings": "Settings",
        "tab_advanced": "Advanced",
        "tab_logs": "Logs",
        "settings_intro_title": "Settings = everyday workflow",
        "settings_intro_body": "Choose where the audio lives, keep a local model selected, set the language, and start. If you are unsure, leave the defaults and use the Fast preset.",
        "settings_step_audio": "1. Audio folder",
        "settings_step_model": "2. Local model",
        "settings_step_start": "3. Start transcription",
        "section_files": "1. Files",
        "label_audio_folder": "Audio folder:",
        "placeholder_audio_folder": "Folder with mp3/wav/flac files",
        "button_browse": "Browse",
        "help_audio_folder": "Put all audio files you want to transcribe into one folder. The app scans the folder and processes every supported file it finds.",
        "label_output_folder": "Output folder:",
        "help_output_folder": "Transcripts are saved here. Each audio file gets a matching .txt output, and optionally a .srt subtitle file.",
        "section_model": "2. Model",
        "label_local_model": "Local model:",
        "button_refresh": "Refresh",
        "button_folder": "Folder",
        "help_model_buttons": "Use Refresh after you add a model to the local models folder. Folder lets you point to a custom local checkpoint directory.",
        "detected_local_models": "Detected local models: {models}",
        "label_transcription_language": "Language:",
        "checkbox_auto_lang": "Auto-detect language",
        "help_language": "Manual language selection is usually more accurate and faster. Turn on auto-detect only when you truly do not know the language in advance.",
        "section_output": "3. Output",
        "checkbox_srt": "Create .srt subtitle files",
        "help_srt": "Enable this if you also want subtitle files for players and video editors. Leave it off when plain text is enough.",
        "adv_intro_title": "Advanced = quality, speed, and memory tuning",
        "adv_intro_body": "Only change these when you know what problem you are solving. For most files the preset and model choice matter more than manual tuning here.",
        "section_compute": "Speed and compute",
        "label_device": "Device:",
        "help_device": "CUDA is fastest on NVIDIA GPUs. CPU is slower but safest. MPS is mainly for Apple Silicon if available.",
        "label_precision": "Precision:",
        "help_precision": "'auto' picks bfloat16 on RTX 30/40/50-series GPUs (most stable, same speed as fp16) and float16 on older CUDA. Force 'float16' if VRAM is very tight on Turing/Volta. 'float32' is safer on CPU or for debugging precision issues.",
        "label_batch_size": "Batch size:",
        "help_batch_size": "Higher batch size can be faster, but it also uses more VRAM. Lower it first if you get memory errors.",
        "label_use_vad": "Use Silero VAD:",
        "checkbox_enable_vad": "Enable VAD",
        "checkbox_enable_vad_missing": "Enable VAD (local repo not found)",
        "vad_hint_path": "Offline VAD path: {path}",
        "vad_hint_repo": "Offline VAD repo: {path}",
        "help_vad": "VAD helps skip silence and cut speech more intelligently. If the local repo is missing, transcription still works, just with simpler chunking.",
        "section_segmentation": "Segmentation",
        "label_vad_threshold": "VAD threshold:",
        "help_vad_threshold": "Higher threshold cuts more aggressively and may remove quiet speech. Lower threshold keeps more audio but may include extra noise.",
        "label_decode_profile": "Decode profile:",
        "decode_balanced": "Balanced",
        "decode_quality": "Quality",
        "help_decode_profile": "Balanced is usually enough. Quality spends more time searching for better wording, which can help on important material.",
        "label_target_db": "Target dB:",
        "help_target_db": "Quiet recordings are normalized toward this level before transcription. The default is a safe middle ground for speech.",
        "label_merge_gap": "Merge gap (sec):",
        "help_merge_gap": "If two speech fragments are separated by only a short pause, this tells the app when to join them back together.",
        "label_min_silence": "Min silence (ms):",
        "help_min_silence": "Smaller values split sooner. Larger values keep more speech together and are often better for natural conversation.",
        "label_chunk_sec": "Chunk sec:",
        "help_chunk_sec": "This is the fallback chunk length when VAD is unavailable or skipped. Longer chunks give more context but use more memory.",
        "label_overlap_sec": "Overlap sec:",
        "help_overlap_sec": "Overlap protects phrase boundaries so words are less likely to be cut between chunks. Larger overlap is safer but slower.",
        "label_max_new_tokens": "Max new tokens:",
        "help_max_new_tokens": "This limits how much text the model can emit for one chunk. Lower values reduce hallucinations; higher values allow longer uninterrupted speech.",
        "logs_intro_title": "Logs explain what the app is doing",
        "logs_intro_body": "If something feels slow or unexpected, this is the first place to look. You will see model loading, segmentation, progress, and errors here.",
        "hardware_check_title": "Hardware Check",
        "hardware_check_body": "NVIDIA GPU was not detected.\n\nAccurate presets will still work, but transcription will run on CPU and be much slower.",
        "cpu_warning_title": "CPU Warning",
        "cpu_warning_body": "Transcription on CPU can be very slow, especially for quality presets.\n\nContinue?",
        "log_config_error": "Config error: {error}",
        "log_audio_folder_not_found": "Audio folder not found: {path}",
        "log_no_audio_files": "No audio files found.",
        "log_found_files": "Found files: {count}. Starting...",
        "log_critical_worker_error": "Critical worker error: {error}",
        "log_stopping": "Stopping...",
        "runtime_model_line": "Model: {model}",
        "runtime_compute_line": "Compute: {compute}",
        "runtime_mode_line": "Mode: offline only",
        "runtime_vad_found_line": "VAD: local repo found",
        "runtime_vad_missing_line": "VAD: optional, local repo missing",
        "model_desc_custom": "Custom local folder selected. Use this when your model is stored outside the built-in cache layout.",
        "model_desc_medium": "Recommended default. Medium is the best balance of speed, memory use, and quality for local work.",
        "model_desc_small": "Faster and lighter, but less accurate on difficult speech or noisy recordings.",
        "model_desc_large": "Highest potential quality, but much heavier. Only use it when the model already exists locally and your hardware can handle it.",
        "model_desc_local_generic": "Local model selected. The app loads it from disk only and does not download anything automatically.",
    },
    "ru": {
        "label_interface_language": "Язык интерфейса",
        "label_preset": "Профиль",
        "preset_fast": "Быстрый",
        "preset_accurate": "Точный",
        "preset_noisy": "Шумная запись",
        "preset_fast_desc": "Лучший режим по умолчанию. Использует более быстрые и безопасные настройки для обычной офлайн-транскрибации.",
        "preset_accurate_desc": "Даёт модели больше контекста и более строгий декодинг, когда точность формулировок важнее скорости.",
        "preset_noisy_desc": "Аккуратнее удерживает речь цельной на шумных записях и при рваных паузах.",
        "subtitle": "Локальное рабочее пространство для транскрибации",
        "button_start": "Начать транскрибацию",
        "button_stop": "Стоп",
        "sidebar_hint": "Если не уверены, оставьте medium, выберите быстрый профиль и начните со вкладки настроек.",
        "status_ready": "Готово",
        "status_done": "Готово",
        "status_error": "Ошибка",
        "status_stopped": "Остановлено",
        "status_processing_file_prefix": "Обработка ",
        "status_processing_percent": "Обработка: {percent}",
        "tab_settings": "Настройки",
        "tab_advanced": "Расширенные",
        "tab_logs": "Логи",
        "settings_intro_title": "Настройки = обычный рабочий сценарий",
        "settings_intro_body": "Выберите папку с аудио, локальную модель, язык транскрибации и запускайте. Если сомневаетесь, оставьте значения по умолчанию и используйте быстрый профиль.",
        "settings_step_audio": "1. Папка с аудио",
        "settings_step_model": "2. Локальная модель",
        "settings_step_start": "3. Запуск",
        "section_files": "1. Файлы",
        "label_audio_folder": "Папка с аудио:",
        "placeholder_audio_folder": "Папка с файлами mp3/wav/flac",
        "button_browse": "Выбрать",
        "help_audio_folder": "Поместите все аудиофайлы для транскрибации в одну папку. Приложение просканирует её и обработает все поддерживаемые файлы.",
        "label_output_folder": "Папка результата:",
        "help_output_folder": "Сюда сохраняются транскрипты. Для каждого аудиофайла создаётся соответствующий .txt, а при желании ещё и .srt.",
        "section_model": "2. Модель",
        "label_local_model": "Локальная модель:",
        "button_refresh": "Обновить",
        "button_folder": "Папка",
        "help_model_buttons": "Нажмите Обновить после добавления модели в локальную папку models. Кнопка Папка позволяет указать собственный каталог с чекпоинтом.",
        "detected_local_models": "Найденные локальные модели: {models}",
        "label_transcription_language": "Язык:",
        "checkbox_auto_lang": "Определять язык автоматически",
        "help_language": "Ручной выбор языка обычно точнее и быстрее. Включайте автоопределение только если заранее не знаете язык записи.",
        "section_output": "3. Вывод",
        "checkbox_srt": "Создавать .srt субтитры",
        "help_srt": "Включите это, если вам нужны субтитры для плееров и видеоредакторов. Отключите, если достаточно обычного текста.",
        "adv_intro_title": "Расширенные = качество, скорость и память",
        "adv_intro_body": "Меняйте эти параметры только если понимаете, какую проблему решаете. Для большинства файлов профиль и выбор модели важнее ручной настройки.",
        "section_compute": "Скорость и вычисления",
        "label_device": "Устройство:",
        "help_device": "CUDA быстрее всего на NVIDIA GPU. CPU медленнее, но надёжнее. MPS в основном нужен для Apple Silicon, если доступен.",
        "label_precision": "Точность:",
        "help_precision": "'auto' выбирает bfloat16 на GPU RTX 30/40/50 (стабильнее, скорость как у fp16) и float16 на более старых CUDA. Принудительно 'float16' стоит ставить только если на Turing/Volta мало VRAM. 'float32' — безопасный режим для CPU и отладки точности.",
        "label_batch_size": "Размер батча:",
        "help_batch_size": "Больший батч может ускорить обработку, но использует больше видеопамяти. Если возникают ошибки памяти, сначала уменьшайте именно его.",
        "label_use_vad": "Использовать Silero VAD:",
        "checkbox_enable_vad": "Включить VAD",
        "checkbox_enable_vad_missing": "Включить VAD (локальный репозиторий не найден)",
        "vad_hint_path": "Локальный путь VAD: {path}",
        "vad_hint_repo": "Локальный репозиторий VAD: {path}",
        "help_vad": "VAD помогает пропускать тишину и умнее разрезать речь. Если локальный репозиторий отсутствует, транскрибация всё равно работает, но с более простой нарезкой.",
        "section_segmentation": "Сегментация",
        "label_vad_threshold": "Порог VAD:",
        "help_vad_threshold": "Более высокий порог режет агрессивнее и может убирать тихую речь. Более низкий порог сохраняет больше аудио, но может пропускать лишний шум.",
        "label_decode_profile": "Профиль декодинга:",
        "decode_balanced": "Сбалансированный",
        "decode_quality": "Качество",
        "help_decode_profile": "Сбалансированного режима обычно достаточно. Режим качества тратит больше времени на поиск лучших формулировок и полезен для важного материала.",
        "label_target_db": "Целевой dB:",
        "help_target_db": "Тихие записи нормализуются к этому уровню перед транскрибацией. Значение по умолчанию является безопасным компромиссом для речи.",
        "label_merge_gap": "Склейка паузы (сек):",
        "help_merge_gap": "Если два фрагмента речи разделены очень короткой паузой, этот параметр определяет, когда приложение должно склеить их обратно.",
        "label_min_silence": "Мин. тишина (мс):",
        "help_min_silence": "Меньшие значения режут раньше. Большие значения лучше сохраняют естественную речь и разговорные фразы целиком.",
        "label_chunk_sec": "Длина чанка (сек):",
        "help_chunk_sec": "Это длина запасного чанка, когда VAD недоступен или отключён. Более длинные чанки дают больше контекста, но требуют больше памяти.",
        "label_overlap_sec": "Перекрытие (сек):",
        "help_overlap_sec": "Перекрытие защищает границы фраз, чтобы слова реже обрезались между чанками. Большее перекрытие безопаснее, но медленнее.",
        "label_max_new_tokens": "Макс. новых токенов:",
        "help_max_new_tokens": "Ограничивает, сколько текста модель может сгенерировать для одного чанка. Меньшие значения снижают галлюцинации, большие позволяют длинную непрерывную речь.",
        "logs_intro_title": "Логи показывают, что делает приложение",
        "logs_intro_body": "Если что-то кажется медленным или странным, сначала смотрите сюда. Здесь видны загрузка модели, сегментация, ход обработки и ошибки.",
        "hardware_check_title": "Проверка оборудования",
        "hardware_check_body": "NVIDIA GPU не обнаружен.\n\nТочные профили всё равно будут работать, но транскрибация пойдёт на CPU и будет заметно медленнее.",
        "cpu_warning_title": "Предупреждение о CPU",
        "cpu_warning_body": "Транскрибация на CPU может быть очень медленной, особенно для профилей качества.\n\nПродолжить?",
        "log_config_error": "Ошибка конфигурации: {error}",
        "log_audio_folder_not_found": "Папка с аудио не найдена: {path}",
        "log_no_audio_files": "Аудиофайлы не найдены.",
        "log_found_files": "Найдено файлов: {count}. Запуск...",
        "log_critical_worker_error": "Критическая ошибка потока: {error}",
        "log_stopping": "Остановка...",
        "runtime_model_line": "Модель: {model}",
        "runtime_compute_line": "Устройство: {compute}",
        "runtime_mode_line": "Режим: только офлайн",
        "runtime_vad_found_line": "VAD: локальный репозиторий найден",
        "runtime_vad_missing_line": "VAD: необязателен, локальный репозиторий не найден",
        "model_desc_custom": "Выбрана пользовательская локальная папка. Используйте этот вариант, если модель хранится вне встроенного кэша.",
        "model_desc_medium": "Рекомендуемый вариант по умолчанию. Medium даёт лучший баланс скорости, памяти и качества для локальной работы.",
        "model_desc_small": "Быстрее и легче, но менее точен на сложной речи и шумных записях.",
        "model_desc_large": "Потенциально самое высокое качество, но модель заметно тяжелее. Используйте её только если она уже есть локально и ваше железо справляется.",
        "model_desc_local_generic": "Выбрана локальная модель. Приложение загружает её только с диска и ничего не скачивает автоматически.",
    },
}


class WhisperGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Whisper Unified")
        self.geometry("1180x820")
        self.minsize(1080, 760)
        self.configure(fg_color="#0E141B")

        self.core = WhisperCore(on_log=self.on_core_log, on_progress=self.on_core_progress)
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.is_running = False
        self.files_to_process: List[Path] = []
        self.ui_language = "ru"
        self.preset_keys = ["fast", "accurate", "noisy"]
        self.decode_profile_keys = ["balanced", "quality"]
        self.localized_help_labels = []
        self.current_status_raw = "Ready"
        self.has_cuda = torch.cuda.is_available()
        self.has_local_vad = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color="#111A24")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Whisper\nUnified",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(28, 6), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=self.t("subtitle"),
            text_color="#8FA7BA",
            font=ctk.CTkFont(size=12),
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
            text_color="#A9BBCB",
            font=ctk.CTkFont(size=12),
            wraplength=220,
            justify="left",
        )
        self.preset_summary_label.grid(row=6, column=0, padx=20, pady=(0, 14), sticky="w")

        self.runtime_summary_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="",
            text_color="#70C7E8",
            font=ctk.CTkFont(size=12),
            wraplength=220,
            justify="left",
        )
        self.runtime_summary_label.grid(row=7, column=0, padx=20, pady=(0, 16), sticky="w")

        self.btn_start = ctk.CTkButton(
            self.sidebar_frame,
            text=self.t("button_start"),
            command=self.start_process,
            fg_color="#2CC985",
            hover_color="#34D894",
            text_color="black",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
        )
        self.btn_start.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.btn_stop = ctk.CTkButton(
            self.sidebar_frame,
            text=self.t("button_stop"),
            command=self.stop_process,
            fg_color="#D63D3D",
            hover_color="#E04B4B",
            state="disabled",
            height=36,
        )
        self.btn_stop.grid(row=9, column=0, padx=20, pady=(0, 14), sticky="ew")

        self.sidebar_hint_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=self.t("sidebar_hint"),
            text_color="#97ABBC",
            font=ctk.CTkFont(size=12),
            wraplength=220,
            justify="left",
        )
        self.sidebar_hint_label.grid(row=10, column=0, padx=20, pady=(0, 18), sticky="w")

        self.progress_bar = ctk.CTkProgressBar(self.sidebar_frame, height=12)
        self.progress_bar.grid(row=11, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.progress_bar.set(0)

        self.lbl_percentage = ctk.CTkLabel(
            self.sidebar_frame,
            text="0%",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.lbl_percentage.grid(row=12, column=0, padx=20, pady=(4, 0), sticky="w")

        self.status_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=self.t("status_ready"),
            wraplength=220,
            text_color="#A6B7C6",
            justify="left",
        )
        self.status_label.grid(row=13, column=0, padx=20, pady=(10, 20), sticky="w")

        self.tabview = ctk.CTkTabview(
            self,
            fg_color="#0E141B",
            segmented_button_fg_color="#16222F",
            segmented_button_selected_color="#21425A",
            segmented_button_selected_hover_color="#2B536F",
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

        self.build_settings_tab()
        self.build_adv_tab()
        self.build_log_tab()

        self.check_hardware()
        self.refresh_local_models()
        self.set_preset_selection("fast")
        self.apply_preset("fast")

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

    def add_help_text(self, parent, row: int, text_key: str, columnspan: int = 3, padx: int = 20):
        label = ctk.CTkLabel(
            parent,
            text=self.t(text_key),
            text_color="#8EA3B5",
            font=ctk.CTkFont(size=12),
            wraplength=760,
            justify="left",
        )
        label.grid(row=row, column=0, columnspan=columnspan, padx=padx, pady=(0, 6), sticky="w")
        self.localized_help_labels.append((label, text_key))
        return label

    def build_settings_tab(self):
        t = self.settings_frame
        t.grid_columnconfigure(1, weight=1)

        self.settings_intro = ctk.CTkFrame(
            t,
            fg_color="#162330",
            border_width=1,
            border_color="#223445",
            corner_radius=18,
        )
        self.settings_intro.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 16))
        self.settings_intro.grid_columnconfigure((0, 1, 2), weight=1)

        self.settings_intro_title_label = ctk.CTkLabel(
            self.settings_intro,
            text=self.t("settings_intro_title"),
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.settings_intro_title_label.grid(row=0, column=0, columnspan=3, padx=18, pady=(16, 4), sticky="w")
        self.settings_intro_body_label = ctk.CTkLabel(
            self.settings_intro,
            text=self.t("settings_intro_body"),
            text_color="#9FB2C5",
            justify="left",
            wraplength=760,
        )
        self.settings_intro_body_label.grid(row=1, column=0, columnspan=3, padx=18, pady=(0, 12), sticky="w")
        self.settings_step_labels = []
        for idx, key in enumerate(["settings_step_audio", "settings_step_model", "settings_step_start"]):
            step_label = ctk.CTkLabel(
                self.settings_intro,
                text=self.t(key),
                fg_color="#203446",
                corner_radius=999,
                padx=12,
                pady=6,
                text_color="#D6E2EB",
            )
            step_label.grid(row=2, column=idx, padx=8, pady=(0, 16), sticky="w")
            self.settings_step_labels.append((step_label, key))

        self.files_section_label = ctk.CTkLabel(t, text=self.t("section_files"), font=ctk.CTkFont(size=16, weight="bold"))
        self.files_section_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))

        self.audio_folder_label = ctk.CTkLabel(t, text=self.t("label_audio_folder"))
        self.audio_folder_label.grid(row=2, column=0, sticky="w", padx=20, pady=5)
        self.entry_audio = ctk.CTkEntry(t, placeholder_text=self.t("placeholder_audio_folder"), height=34)
        self.entry_audio.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        self.entry_audio.insert(0, str(Path("./audio").absolute()))
        self.btn_browse_audio = ctk.CTkButton(t, text=self.t("button_browse"), width=86, command=self.browse_audio)
        self.btn_browse_audio.grid(row=2, column=2, padx=20)
        self.add_help_text(
            t,
            3,
            "help_audio_folder",
        )

        self.output_folder_label = ctk.CTkLabel(t, text=self.t("label_output_folder"))
        self.output_folder_label.grid(row=4, column=0, sticky="w", padx=20, pady=5)
        self.entry_output = ctk.CTkEntry(t, height=34)
        self.entry_output.grid(row=4, column=1, sticky="ew", padx=10, pady=5)
        self.entry_output.insert(0, str(Path("./audio_to_text").absolute()))
        self.btn_browse_output = ctk.CTkButton(t, text=self.t("button_browse"), width=86, command=self.browse_output)
        self.btn_browse_output.grid(row=4, column=2, padx=20)
        self.add_help_text(
            t,
            5,
            "help_output_folder",
        )

        self.model_section_label = ctk.CTkLabel(t, text=self.t("section_model"), font=ctk.CTkFont(size=16, weight="bold"))
        self.model_section_label.grid(row=6, column=0, sticky="w", padx=10, pady=(20, 5))

        self.local_model_label = ctk.CTkLabel(t, text=self.t("label_local_model"))
        self.local_model_label.grid(row=7, column=0, sticky="w", padx=20, pady=5)
        self.combo_model = ctk.CTkComboBox(
            t,
            values=["openai/whisper-medium"],
            height=34,
            command=lambda _value: self.refresh_runtime_summary(),
        )
        self.combo_model.grid(row=7, column=1, sticky="ew", padx=10, pady=5)
        self.combo_model.set("openai/whisper-medium")
        self.model_button_frame = ctk.CTkFrame(t, fg_color="transparent")
        self.model_button_frame.grid(row=7, column=2, sticky="e", padx=10)
        self.btn_refresh_models = ctk.CTkButton(
            self.model_button_frame,
            text=self.t("button_refresh"),
            width=82,
            command=self.refresh_local_models,
        )
        self.btn_refresh_models.pack(side="left", padx=(0, 6))
        self.btn_model_folder = ctk.CTkButton(
            self.model_button_frame,
            text=self.t("button_folder"),
            width=82,
            command=self.browse_model_folder,
        )
        self.btn_model_folder.pack(side="left")

        self.model_help_label = ctk.CTkLabel(
            t,
            text=self.describe_model("openai/whisper-medium"),
            text_color="#6FC5E7",
            justify="left",
            wraplength=760,
        )
        self.model_help_label.grid(row=8, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 4))
        self.local_models_note = ctk.CTkLabel(
            t,
            text="",
            text_color="#9BB0C4",
            justify="left",
            wraplength=760,
        )
        self.local_models_note.grid(row=9, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 2))
        self.add_help_text(
            t,
            10,
            "help_model_buttons",
        )

        self.transcription_language_label = ctk.CTkLabel(t, text=self.t("label_transcription_language"))
        self.transcription_language_label.grid(row=11, column=0, sticky="w", padx=20, pady=5)
        self.combo_lang = ctk.CTkComboBox(t, values=["ru", "en", "de", "fr", "es", "it", "ja", "zh"])
        self.combo_lang.grid(row=11, column=1, sticky="w", padx=10, pady=5)
        self.combo_lang.set("ru")

        self.check_auto_lang = ctk.CTkCheckBox(t, text=self.t("checkbox_auto_lang"))
        self.check_auto_lang.grid(row=11, column=2, sticky="w", padx=10, pady=5)
        self.add_help_text(
            t,
            12,
            "help_language",
        )

        self.output_section_label = ctk.CTkLabel(t, text=self.t("section_output"), font=ctk.CTkFont(size=16, weight="bold"))
        self.output_section_label.grid(row=13, column=0, sticky="w", padx=10, pady=(20, 5))

        self.check_srt = ctk.CTkCheckBox(t, text=self.t("checkbox_srt"))
        self.check_srt.grid(row=14, column=1, sticky="w", padx=10, pady=(4, 5))
        self.add_help_text(
            t,
            15,
            "help_srt",
        )

    def build_adv_tab(self):
        t = self.adv_frame
        t.grid_columnconfigure(1, weight=1)

        self.adv_intro = ctk.CTkFrame(
            t,
            fg_color="#181F28",
            border_width=1,
            border_color="#2A3C4C",
            corner_radius=18,
        )
        self.adv_intro.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 16))
        self.adv_intro_title_label = ctk.CTkLabel(
            self.adv_intro,
            text=self.t("adv_intro_title"),
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.adv_intro_title_label.grid(row=0, column=0, padx=18, pady=(16, 4), sticky="w")
        self.adv_intro_body_label = ctk.CTkLabel(
            self.adv_intro,
            text=self.t("adv_intro_body"),
            text_color="#D7B16E",
            justify="left",
            wraplength=760,
        )
        self.adv_intro_body_label.grid(row=1, column=0, padx=18, pady=(0, 16), sticky="w")

        self.compute_section_label = ctk.CTkLabel(t, text=self.t("section_compute"), font=ctk.CTkFont(size=16, weight="bold"))
        self.compute_section_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))

        self.device_label = ctk.CTkLabel(t, text=self.t("label_device"))
        self.device_label.grid(row=2, column=0, sticky="w", padx=20, pady=5)
        self.combo_device = ctk.CTkComboBox(
            t,
            values=["cuda", "cpu", "mps"],
            height=34,
            command=lambda _value: self.refresh_runtime_summary(),
        )
        self.combo_device.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        self.add_help_text(
            t,
            3,
            "help_device",
        )

        self.precision_label = ctk.CTkLabel(t, text=self.t("label_precision"))
        self.precision_label.grid(row=4, column=0, sticky="w", padx=20, pady=5)
        self.combo_dtype = ctk.CTkComboBox(
            t,
            values=["auto", "bfloat16", "float16", "float32"],
            height=34,
            command=lambda _value: self.refresh_runtime_summary(),
        )
        self.combo_dtype.grid(row=4, column=1, sticky="ew", padx=10, pady=5)
        self.add_help_text(
            t,
            5,
            "help_precision",
        )

        self.batch_size_label = ctk.CTkLabel(t, text=self.t("label_batch_size"))
        self.batch_size_label.grid(row=6, column=0, sticky="w", padx=20, pady=5)
        self.entry_batch = ctk.CTkEntry(t, height=34)
        self.entry_batch.grid(row=6, column=1, sticky="ew", padx=10, pady=5)
        self.entry_batch.insert(0, "2")
        self.add_help_text(
            t,
            7,
            "help_batch_size",
        )

        self.use_vad_label = ctk.CTkLabel(t, text=self.t("label_use_vad"))
        self.use_vad_label.grid(row=8, column=0, sticky="w", padx=20, pady=5)
        self.check_vad = ctk.CTkCheckBox(t, text=self.t("checkbox_enable_vad"))
        self.check_vad.grid(row=8, column=1, sticky="w", padx=10, pady=5)
        self.check_vad.select()
        self.vad_hint_label = ctk.CTkLabel(
            t,
            text=self.t("vad_hint_path", path="models\\silero-vad"),
            text_color="#9BB0C4",
            justify="left",
            wraplength=300,
        )
        self.vad_hint_label.grid(row=8, column=2, sticky="w", padx=10, pady=5)
        self.add_help_text(
            t,
            9,
            "help_vad",
        )

        self.segmentation_section_label = ctk.CTkLabel(t, text=self.t("section_segmentation"), font=ctk.CTkFont(size=16, weight="bold"))
        self.segmentation_section_label.grid(row=10, column=0, sticky="w", padx=10, pady=(20, 5))

        self.vad_threshold_label = ctk.CTkLabel(t, text=self.t("label_vad_threshold"))
        self.vad_threshold_label.grid(row=11, column=0, sticky="w", padx=20, pady=5)
        self.slider_frame = ctk.CTkFrame(t, fg_color="transparent")
        self.slider_frame.grid(row=11, column=1, sticky="ew")
        self.slider_frame.grid_columnconfigure(0, weight=1)

        self.slider_vad = ctk.CTkSlider(
            self.slider_frame,
            from_=0.1,
            to=0.9,
            number_of_steps=8,
            command=self.update_vad_label,
        )
        self.slider_vad.grid(row=0, column=0, sticky="ew", padx=(10, 5), pady=5)
        self.slider_vad.set(0.5)

        self.lbl_vad_val = ctk.CTkLabel(
            self.slider_frame,
            text="0.5",
            width=40,
            font=ctk.CTkFont(weight="bold"),
        )
        self.lbl_vad_val.grid(row=0, column=1, padx=5)
        self.add_help_text(
            t,
            12,
            "help_vad_threshold",
        )

        self.decode_profile_label_widget = ctk.CTkLabel(t, text=self.t("label_decode_profile"))
        self.decode_profile_label_widget.grid(row=13, column=0, sticky="w", padx=20, pady=5)
        self.combo_decode = ctk.CTkComboBox(
            t,
            values=self.localized_decode_profile_values(),
            height=34,
            command=lambda _value: self.refresh_runtime_summary(),
        )
        self.combo_decode.grid(row=13, column=1, sticky="ew", padx=10, pady=5)
        self.set_decode_profile_selection("balanced")
        self.add_help_text(
            t,
            14,
            "help_decode_profile",
        )

        self.target_db_label = ctk.CTkLabel(t, text=self.t("label_target_db"))
        self.target_db_label.grid(row=15, column=0, sticky="w", padx=20, pady=5)
        self.entry_target_db = ctk.CTkEntry(t, height=34)
        self.entry_target_db.grid(row=15, column=1, sticky="ew", padx=10, pady=5)
        self.entry_target_db.insert(0, "-20.0")
        self.add_help_text(
            t,
            16,
            "help_target_db",
        )

        self.merge_gap_label = ctk.CTkLabel(t, text=self.t("label_merge_gap"))
        self.merge_gap_label.grid(row=17, column=0, sticky="w", padx=20, pady=5)
        self.entry_vad_merge_gap = ctk.CTkEntry(t, height=34)
        self.entry_vad_merge_gap.grid(row=17, column=1, sticky="ew", padx=10, pady=5)
        self.entry_vad_merge_gap.insert(0, "0.25")
        self.add_help_text(
            t,
            18,
            "help_merge_gap",
        )

        self.min_silence_label = ctk.CTkLabel(t, text=self.t("label_min_silence"))
        self.min_silence_label.grid(row=19, column=0, sticky="w", padx=20, pady=5)
        self.entry_vad_silence = ctk.CTkEntry(t, height=34)
        self.entry_vad_silence.grid(row=19, column=1, sticky="ew", padx=10, pady=5)
        self.entry_vad_silence.insert(0, "100")
        self.add_help_text(
            t,
            20,
            "help_min_silence",
        )

        self.chunk_sec_label = ctk.CTkLabel(t, text=self.t("label_chunk_sec"))
        self.chunk_sec_label.grid(row=21, column=0, sticky="w", padx=20, pady=5)
        self.entry_chunk_sec = ctk.CTkEntry(t, height=34)
        self.entry_chunk_sec.grid(row=21, column=1, sticky="ew", padx=10, pady=5)
        self.entry_chunk_sec.insert(0, "20.0")
        self.add_help_text(
            t,
            22,
            "help_chunk_sec",
        )

        self.overlap_sec_label = ctk.CTkLabel(t, text=self.t("label_overlap_sec"))
        self.overlap_sec_label.grid(row=23, column=0, sticky="w", padx=20, pady=5)
        self.entry_overlap_sec = ctk.CTkEntry(t, height=34)
        self.entry_overlap_sec.grid(row=23, column=1, sticky="ew", padx=10, pady=5)
        self.entry_overlap_sec.insert(0, "3.0")
        self.add_help_text(
            t,
            24,
            "help_overlap_sec",
        )

        self.max_new_tokens_label = ctk.CTkLabel(t, text=self.t("label_max_new_tokens"))
        self.max_new_tokens_label.grid(row=25, column=0, sticky="w", padx=20, pady=5)
        self.entry_max_tokens = ctk.CTkEntry(t, height=34)
        self.entry_max_tokens.grid(row=25, column=1, sticky="ew", padx=10, pady=5)
        self.entry_max_tokens.insert(0, "225")
        self.add_help_text(
            t,
            26,
            "help_max_new_tokens",
        )

    def build_log_tab(self):
        t = self.tab_logs
        t.grid_rowconfigure(1, weight=1)
        t.grid_columnconfigure(0, weight=1)

        self.logs_info_frame = ctk.CTkFrame(
            t,
            fg_color="#161F29",
            border_width=1,
            border_color="#223445",
            corner_radius=16,
        )
        self.logs_info_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 10))
        self.logs_intro_title_label = ctk.CTkLabel(
            self.logs_info_frame,
            text=self.t("logs_intro_title"),
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.logs_intro_title_label.grid(row=0, column=0, padx=16, pady=(14, 4), sticky="w")
        self.logs_intro_body_label = ctk.CTkLabel(
            self.logs_info_frame,
            text=self.t("logs_intro_body"),
            text_color="#97ABBC",
            wraplength=820,
            justify="left",
        )
        self.logs_intro_body_label.grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")

        self.txt_log = ctk.CTkTextbox(t, state="disabled", font=("Consolas", 12))
        self.txt_log.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def set_entry_value(self, entry: ctk.CTkEntry, value):
        entry.delete(0, "end")
        entry.insert(0, str(value))

    def discover_local_models(self) -> List[str]:
        models_root = Path(__file__).parent / "models"
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
            choices = ["openai/whisper-medium"]

        current_value = self.combo_model.get().strip() if hasattr(self, "combo_model") else ""
        if current_value and current_value not in choices and Path(current_value).exists():
            choices.append(current_value)
        self.combo_model.configure(values=choices)

        if current_value in choices:
            self.combo_model.set(current_value)
        elif "openai/whisper-medium" in choices:
            self.combo_model.set("openai/whisper-medium")
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
        preset_key = choice if choice in self.preset_keys else self.preset_key_from_value(choice)
        use_cuda = self.has_cuda

        def set_vad_enabled(enabled: bool):
            if enabled and self.has_local_vad:
                self.check_vad.select()
            else:
                self.check_vad.deselect()

        if preset_key == "fast":
            self.combo_device.set("cuda" if use_cuda else "cpu")
            # "auto" lets WhisperCore pick bf16 on Ampere+ (cc>=8), fp16 on older
            # CUDA, fp32 on CPU — bf16 is more numerically stable for Whisper.
            self.combo_dtype.set("auto" if use_cuda else "float32")
            self.set_entry_value(self.entry_batch, 4 if use_cuda else 1)
            self.set_decode_profile_selection("balanced")
            self.set_entry_value(self.entry_target_db, -20.0)
            self.set_entry_value(self.entry_vad_merge_gap, 0.15)
            self.set_entry_value(self.entry_vad_silence, 100)
            self.set_entry_value(self.entry_chunk_sec, 12.0)
            self.set_entry_value(self.entry_overlap_sec, 1.5)
            self.set_entry_value(self.entry_max_tokens, 160)
            set_vad_enabled(True)
            self.slider_vad.set(0.5)
            self.update_vad_label(0.5)
        elif preset_key == "accurate":
            self.combo_device.set("cuda" if use_cuda else "cpu")
            # "auto" lets WhisperCore pick bf16 on Ampere+ (cc>=8), fp16 on older
            # CUDA, fp32 on CPU — bf16 is more numerically stable for Whisper.
            self.combo_dtype.set("auto" if use_cuda else "float32")
            self.set_entry_value(self.entry_batch, 2 if use_cuda else 1)
            self.set_decode_profile_selection("quality")
            self.set_entry_value(self.entry_target_db, -20.0)
            self.set_entry_value(self.entry_vad_merge_gap, 0.25)
            self.set_entry_value(self.entry_vad_silence, 140)
            self.set_entry_value(self.entry_chunk_sec, 20.0)
            self.set_entry_value(self.entry_overlap_sec, 3.0)
            self.set_entry_value(self.entry_max_tokens, 225)
            set_vad_enabled(True)
            self.slider_vad.set(0.45)
            self.update_vad_label(0.45)
        elif preset_key == "noisy":
            self.combo_device.set("cuda" if use_cuda else "cpu")
            # "auto" lets WhisperCore pick bf16 on Ampere+ (cc>=8), fp16 on older
            # CUDA, fp32 on CPU — bf16 is more numerically stable for Whisper.
            self.combo_dtype.set("auto" if use_cuda else "float32")
            self.set_entry_value(self.entry_batch, 2 if use_cuda else 1)
            self.set_decode_profile_selection("quality")
            self.set_entry_value(self.entry_target_db, -20.0)
            self.set_entry_value(self.entry_vad_merge_gap, 0.45)
            self.set_entry_value(self.entry_vad_silence, 250)
            self.set_entry_value(self.entry_chunk_sec, 22.0)
            self.set_entry_value(self.entry_overlap_sec, 3.0)
            self.set_entry_value(self.entry_max_tokens, 225)
            set_vad_enabled(True)
            self.slider_vad.set(0.35)
            self.update_vad_label(0.35)
        self.refresh_runtime_summary()

    def update_vad_label(self, val):
        self.lbl_vad_val.configure(text=f"{float(val):.2f}")

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

        exts = ["*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg"]
        self.files_to_process = []
        for ext in exts:
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
