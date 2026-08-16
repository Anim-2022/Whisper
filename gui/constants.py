"""Single source of truth for GUI constants.

Anything that used to live as a magic string/number inside whisper_gui.py
should be defined here so that:
  - presets, validators, and tests share the same lists
  - changing a color or default value touches one place
  - new modules (settings_store, validators) can import without re-typing values
"""
from pathlib import Path

from whisper_engine.audio import AUDIO_SUFFIXES
from whisper_engine.device import COMPUTE_TYPES as _COMPUTE_TYPES
from whisper_engine.device import DEVICES as _DEVICES

# --------------------------------------------------------------------------
# Theming and window geometry
# --------------------------------------------------------------------------
APPEARANCE_MODE: str = "Dark"          # "Dark" | "Light" | "System"
COLOR_THEME: str = "blue"              # built-in customtkinter theme

WINDOW_TITLE: str = "Whisper Unified"
WINDOW_GEOMETRY: str = "1180x820"
WINDOW_MIN_SIZE: tuple[int, int] = (1080, 760)

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
TRANSCRIPTION_LANGUAGES: list[str] = [
    "ru", "en", "de", "fr", "es", "it", "ja", "zh",
]
# CTranslate2 has no MPS backend, so "mps" is gone (it never worked on Windows anyway).
DEVICES: list[str] = list(_DEVICES)
COMPUTE_TYPES: list[str] = list(_COMPUTE_TYPES)

# Audio discovery is a suffix set owned by whisper_engine.audio, not a glob list:
# PyAV decides what is actually decodable, and one iterdir pass cannot enqueue a
# file twice the way overlapping globs could.
AUDIO_SUFFIXES = AUDIO_SUFFIXES

OUTPUT_FORMATS: list[str] = ["txt", "srt", "vtt", "json", "md"]
DEFAULT_OUTPUT_FORMATS: tuple[str, ...] = ("txt",)

# Order of presets in the UI combo and used as canonical keys everywhere
PRESET_KEYS: list[str] = ["fast", "accurate", "noisy"]
# Replaces the old balanced/quality decode profiles. The split is now honest about
# what actually differs: the batched pipeline discards condition-on-previous-text,
# the temperature ladder and the hallucination filter.
MODE_KEYS: list[str] = ["batched", "sequential"]

# UI-language combo (interface language, separate from transcription language)
UI_LANGUAGES: list[str] = ["ru", "en"]
UI_LANGUAGE_LABELS: dict[str, str] = {"ru": "Русский", "en": "English"}
DEFAULT_UI_LANGUAGE: str = "ru"

# --------------------------------------------------------------------------
# Defaults that touch the filesystem
# --------------------------------------------------------------------------
# Name of a directory under models/ct2/, produced by tools/convert_models.py.
#
# Measured on ~3 h Russian lecture audio with English technical terms: medium in
# batched mode runs at ~60-70x realtime and matches or beats the previous engine
# on punctuation density and on retention of English terms. large-v3-turbo is
# faster still, but it is distilled to four decoder layers and loses punctuation
# badly (one test file came out with none at all) and transliterates technical
# terms, so it stays an option rather than the default.
DEFAULT_MODEL: str = "whisper-medium"

# These paths are evaluated relative to whisper_gui.py's directory at startup;
# the GUI converts them to absolute. settings_store (Phase B) will override.
DEFAULT_AUDIO_DIR_REL: str = "./audio"
DEFAULT_OUTPUT_DIR_REL: str = "./audio_to_text"
LOCAL_MODELS_SUBDIR: str = "models"      # under whisper_gui.py's parent

# --------------------------------------------------------------------------
# Validation ranges (used in Phase D, but defined here as single source of truth)
# --------------------------------------------------------------------------
# chunk_sec / overlap_sec / target_db / max_new_tokens / vad_merge_gap are gone:
# faster-whisper uses fixed 30 s windows, seeks on decoded timestamps, and merges
# speech by silence duration, so none of those knobs exist any more.
BATCH_SIZE_RANGE: tuple[int, int] = (1, 32)
BEAM_SIZE_RANGE: tuple[int, int] = (1, 10)
VAD_THRESHOLD_RANGE: tuple[float, float] = (0.1, 0.9)
VAD_MIN_SPEECH_MS_RANGE: tuple[int, int] = (0, 1000)
VAD_MIN_SILENCE_MS_RANGE: tuple[int, int] = (0, 3000)
VAD_SPEECH_PAD_MS_RANGE: tuple[int, int] = (0, 1000)
NO_SPEECH_THRESHOLD_RANGE: tuple[float, float] = (0.0, 1.0)
COMPRESSION_RATIO_RANGE: tuple[float, float] = (1.0, 5.0)
LOG_PROB_THRESHOLD_RANGE: tuple[float, float] = (-5.0, 0.0)
HALLUCINATION_SILENCE_RANGE: tuple[float, float] = (0.0, 10.0)

# --------------------------------------------------------------------------
# Presets, re-authored for faster-whisper.
#
# "accurate" is the recommended default: batched medium measured ~60-70x realtime
# on a 16 GB GPU while matching the previous engine's punctuation. There is no
# longer a meaningful speed/quality trade-off between it and "fast", only a beam
# width, so "fast" exists for drafts and slow machines.
#
# "noisy" is the only sequential preset. Sequential is what unlocks the temperature
# ladder, condition-on-previous-text and the hallucination filter — all of which
# the batched pipeline discards — at roughly a quarter of the speed.
#
# Schema per preset (batch_size_cuda/cpu are applied by the GUI based on device):
#   batched, batch_size_cuda, batch_size_cpu, beam_size, temperature_fallback,
#   condition_on_previous_text, no_speech_threshold, compression_ratio_threshold,
#   log_prob_threshold, hallucination_silence_threshold, word_timestamps,
#   vad_enabled, vad_threshold, vad_min_speech_ms, vad_min_silence_ms,
#   vad_speech_pad_ms
# --------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "fast": {
        "batched": True,
        "batch_size_cuda": 16,
        "batch_size_cpu": 4,
        "beam_size": 1,
        "temperature_fallback": False,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "hallucination_silence_threshold": None,
        "word_timestamps": False,
        "vad_enabled": True,
        "vad_threshold": 0.50,
        "vad_min_speech_ms": 250,
        "vad_min_silence_ms": 500,
        "vad_speech_pad_ms": 200,
    },
    "accurate": {
        "batched": True,
        "batch_size_cuda": 16,
        "batch_size_cpu": 4,
        "beam_size": 5,
        "temperature_fallback": False,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "hallucination_silence_threshold": None,
        "word_timestamps": False,
        "vad_enabled": True,
        "vad_threshold": 0.45,
        "vad_min_speech_ms": 250,
        "vad_min_silence_ms": 700,
        "vad_speech_pad_ms": 400,
    },
    "noisy": {
        "batched": False,
        "batch_size_cuda": 8,
        "batch_size_cpu": 1,
        "beam_size": 5,
        "temperature_fallback": True,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.5,
        "compression_ratio_threshold": 2.2,
        "log_prob_threshold": -0.8,
        "hallucination_silence_threshold": 1.5,
        "word_timestamps": True,
        "vad_enabled": True,
        "vad_threshold": 0.35,
        "vad_min_speech_ms": 250,
        "vad_min_silence_ms": 1000,
        "vad_speech_pad_ms": 600,
    },
}

DEFAULT_PRESET: str = "accurate"

# Preset keys that map straight onto TranscriptionConfig fields of the same name.
_PRESET_PASSTHROUGH: tuple[str, ...] = (
    "batched", "beam_size", "temperature_fallback", "condition_on_previous_text",
    "no_speech_threshold", "compression_ratio_threshold", "log_prob_threshold",
    "hallucination_silence_threshold", "word_timestamps",
    "vad_threshold", "vad_min_speech_ms", "vad_min_silence_ms", "vad_speech_pad_ms",
)


def preset_to_config_kwargs(preset: dict, has_cuda: bool = True) -> dict:
    """Translate a preset into TranscriptionConfig keyword arguments.

    Keeps the CUDA/CPU batch-size split in one place instead of duplicating it in
    the GUI and the CLI.
    """
    kwargs = {key: preset[key] for key in _PRESET_PASSTHROUGH if key in preset}
    kwargs["batch_size"] = preset["batch_size_cuda" if has_cuda else "batch_size_cpu"]
    kwargs["vad_filter"] = preset.get("vad_enabled", True)
    return kwargs


# --------------------------------------------------------------------------
# Per-field defaults shown when no preset has been applied yet
# --------------------------------------------------------------------------
DEFAULTS: dict[str, object] = {
    **{k: v for k, v in PRESETS[DEFAULT_PRESET].items()
       if k not in ("batch_size_cuda", "batch_size_cpu")},
    "batch_size": PRESETS[DEFAULT_PRESET]["batch_size_cuda"],
    "transcription_lang": "ru",
    "initial_prompt": "",
    "formats": list(DEFAULT_OUTPUT_FORMATS),
}


def resolve_default_audio_dir(base: Path) -> Path:
    """Return absolute default audio dir given the app's base directory."""
    return (base / DEFAULT_AUDIO_DIR_REL).resolve()


def resolve_default_output_dir(base: Path) -> Path:
    """Return absolute default output dir given the app's base directory."""
    return (base / DEFAULT_OUTPUT_DIR_REL).resolve()
