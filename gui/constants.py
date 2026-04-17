# -*- coding: utf-8 -*-
"""Single source of truth for GUI constants.

Anything that used to live as a magic string/number inside whisper_gui.py
should be defined here so that:
  - presets, validators, and tests share the same lists
  - changing a color or default value touches one place
  - new modules (settings_store, validators) can import without re-typing values
"""
from pathlib import Path
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------
# Theming and window geometry
# --------------------------------------------------------------------------
APPEARANCE_MODE: str = "Dark"          # "Dark" | "Light" | "System"
COLOR_THEME: str = "blue"              # built-in customtkinter theme

WINDOW_TITLE: str = "Whisper Unified"
WINDOW_GEOMETRY: str = "1180x820"
WINDOW_MIN_SIZE: Tuple[int, int] = (1080, 760)

SIDEBAR_WIDTH: int = 260
WRAPLENGTH_SIDEBAR: int = 220
WRAPLENGTH_BODY: int = 760
WRAPLENGTH_LOGS_INTRO: int = 820
WRAPLENGTH_VAD_HINT: int = 300

# Help-label font size (used by add_help_text)
HELP_FONT_SIZE: int = 12
HEADER_FONT_SIZE: int = 16
SECTION_INTRO_TITLE_SIZE: int = 18
LOGO_FONT_SIZE: int = 28

# --------------------------------------------------------------------------
# Color palette — dark-only. We used to ship (light, dark) tuples, but
# the light variant never felt polished (banner stays dark, Consolas-on-
# cream looks off) and carrying two palettes just to offer a toggle was
# not worth the surface area. appearance_mode is now pinned to "Dark".
# --------------------------------------------------------------------------
COLOR_BG: str = "#0E141B"              # main window
COLOR_SIDEBAR_BG: str = "#111A24"      # left panel
COLOR_PANEL_BG: str = "#162330"        # settings intro card
COLOR_PANEL_BORDER: str = "#223445"
COLOR_ADV_PANEL_BG: str = "#181F28"
COLOR_ADV_PANEL_BORDER: str = "#2A3C4C"
COLOR_LOG_PANEL_BG: str = "#161F29"
COLOR_TAB_BG: str = "#16222F"
COLOR_TAB_SELECTED: str = "#21425A"
COLOR_TAB_SELECTED_HOVER: str = "#2B536F"
COLOR_STEP_BADGE_BG: str = "#203446"
COLOR_STEP_BADGE_TEXT: str = "#D6E2EB"

COLOR_TEXT_SUBTITLE: str = "#8FA7BA"
COLOR_TEXT_PRESET_SUMMARY: str = "#A9BBCB"
COLOR_TEXT_RUNTIME: str = "#70C7E8"
COLOR_TEXT_HINT: str = "#97ABBC"
COLOR_TEXT_STATUS: str = "#A6B7C6"
COLOR_TEXT_HELP: str = "#8EA3B5"
COLOR_TEXT_INTRO_BODY: str = "#9FB2C5"
COLOR_TEXT_MODEL_HELP: str = "#6FC5E7"
COLOR_TEXT_LOCAL_NOTE: str = "#9BB0C4"
COLOR_TEXT_ADV_INTRO: str = "#D7B16E"  # warning-amber for "advanced = caution"

COLOR_BUTTON_START_FG: str = "#2CC985"
COLOR_BUTTON_START_HOVER: str = "#34D894"
COLOR_BUTTON_START_TEXT: str = "black"
COLOR_BUTTON_STOP_FG: str = "#D63D3D"
COLOR_BUTTON_STOP_HOVER: str = "#E04B4B"

# --------------------------------------------------------------------------
# Lists used by combos and file scanning
# --------------------------------------------------------------------------
TRANSCRIPTION_LANGUAGES: List[str] = [
    "ru", "en", "de", "fr", "es", "it", "ja", "zh",
]
DEVICES: List[str] = ["cuda", "cpu", "mps"]
DTYPES: List[str] = ["auto", "bfloat16", "float16", "float32"]
AUDIO_EXTENSIONS: Tuple[str, ...] = (
    "*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg",
)

# Order of presets in the UI combo and used as canonical keys everywhere
PRESET_KEYS: List[str] = ["fast", "accurate", "noisy"]
DECODE_PROFILE_KEYS: List[str] = ["balanced", "quality"]

# UI-language combo (interface language, separate from transcription language)
UI_LANGUAGES: List[str] = ["ru", "en"]
UI_LANGUAGE_LABELS: Dict[str, str] = {"ru": "Русский", "en": "English"}
DEFAULT_UI_LANGUAGE: str = "ru"

# --------------------------------------------------------------------------
# Defaults that touch the filesystem
# --------------------------------------------------------------------------
DEFAULT_MODEL: str = "openai/whisper-medium"

# These paths are evaluated relative to whisper_gui.py's directory at startup;
# the GUI converts them to absolute. settings_store (Phase B) will override.
DEFAULT_AUDIO_DIR_REL: str = "./audio"
DEFAULT_OUTPUT_DIR_REL: str = "./audio_to_text"
LOCAL_MODELS_SUBDIR: str = "models"      # under whisper_gui.py's parent

# --------------------------------------------------------------------------
# Validation ranges (used in Phase D, but defined here as single source of truth)
# --------------------------------------------------------------------------
MAX_NEW_TOKENS_RANGE: Tuple[int, int] = (32, 448)   # Whisper hard cap is 448
BATCH_SIZE_RANGE: Tuple[int, int] = (1, 32)
CHUNK_SEC_RANGE: Tuple[float, float] = (6.0, 30.0)
OVERLAP_SEC_RANGE: Tuple[float, float] = (0.0, 6.0)
TARGET_DB_RANGE: Tuple[float, float] = (-40.0, 0.0)
VAD_THRESHOLD_RANGE: Tuple[float, float] = (0.1, 0.9)
VAD_MIN_SILENCE_MS_RANGE: Tuple[int, int] = (50, 2000)
VAD_MERGE_GAP_SEC_RANGE: Tuple[float, float] = (0.0, 2.0)

# --------------------------------------------------------------------------
# Presets — full data (used to live inside apply_preset as magic numbers).
# Each preset says: pick CUDA when available, "auto" dtype when CUDA available.
# Per-field tuples are ints/floats only; the GUI applies them.
#
# Schema (per preset):
#   batch_size_cuda / batch_size_cpu : int
#   decode_profile  : str  (one of DECODE_PROFILE_KEYS)
#   target_db       : float
#   vad_merge_gap   : float
#   vad_silence_ms  : int
#   chunk_sec       : float
#   overlap_sec     : float
#   max_new_tokens  : int
#   vad_enabled     : bool
#   vad_threshold   : float
# --------------------------------------------------------------------------
PRESETS: Dict[str, Dict] = {
    "fast": {
        "batch_size_cuda": 4,
        "batch_size_cpu": 1,
        "decode_profile": "balanced",
        "target_db": -20.0,
        "vad_merge_gap": 0.15,
        "vad_silence_ms": 100,
        "chunk_sec": 12.0,
        "overlap_sec": 1.5,
        "max_new_tokens": 128,
        "vad_enabled": True,
        "vad_threshold": 0.50,
    },
    "accurate": {
        "batch_size_cuda": 2,
        "batch_size_cpu": 1,
        "decode_profile": "quality",
        "target_db": -20.0,
        "vad_merge_gap": 0.25,
        "vad_silence_ms": 140,
        "chunk_sec": 20.0,
        "overlap_sec": 3.0,
        "max_new_tokens": 160,
        "vad_enabled": True,
        "vad_threshold": 0.45,
    },
    "noisy": {
        "batch_size_cuda": 2,
        "batch_size_cpu": 1,
        "decode_profile": "quality",
        "target_db": -20.0,
        "vad_merge_gap": 0.45,
        "vad_silence_ms": 250,
        "chunk_sec": 22.0,
        "overlap_sec": 3.0,
        "max_new_tokens": 192,
        "vad_enabled": True,
        "vad_threshold": 0.35,
    },
}

# --------------------------------------------------------------------------
# Per-field defaults shown when no preset has been applied yet
# (these mirror the initial Entry.insert calls in build_adv_tab)
# --------------------------------------------------------------------------
DEFAULTS: Dict[str, object] = {
    "batch_size": 2,
    "target_db": -20.0,
    "vad_merge_gap": 0.25,
    "vad_silence_ms": 100,
    "chunk_sec": 20.0,
    "overlap_sec": 3.0,
    "max_new_tokens": 128,
    "vad_threshold": 0.5,
    "decode_profile": "balanced",
    "transcription_lang": "ru",
}


def resolve_default_audio_dir(base: Path) -> Path:
    """Return absolute default audio dir given the app's base directory."""
    return (base / DEFAULT_AUDIO_DIR_REL).resolve()


def resolve_default_output_dir(base: Path) -> Path:
    """Return absolute default output dir given the app's base directory."""
    return (base / DEFAULT_OUTPUT_DIR_REL).resolve()
