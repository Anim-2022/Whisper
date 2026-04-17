# -*- coding: utf-8 -*-
"""Persist user settings between sessions.

Layout on disk:
    %USERPROFILE%/.whisper_gui/settings.json   (Windows)
    ~/.whisper_gui/settings.json               (Linux/macOS)

Design notes:
- Atomic write via tmp + os.replace so a crash mid-write never leaves a
  half-written JSON that breaks startup.
- load() never raises: malformed file -> log + return defaults. The GUI
  must always start.
- Schema is a flat dict[str, scalar]. Adding a key is backward-compatible
  (old files don't have it -> default). Removing one is too (extra keys
  are ignored).
- The GUI is the source of truth for *current* state; settings_store just
  remembers the last-known good snapshot.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from . import constants as C


SETTINGS_DIRNAME = ".whisper_gui"
SETTINGS_FILENAME = "settings.json"
SCHEMA_VERSION = 1


def get_default_path() -> Path:
    """Return absolute path to the settings file under the user's home dir."""
    return Path.home() / SETTINGS_DIRNAME / SETTINGS_FILENAME


# Defaults for any missing key. Matches the values the app would show on a
# brand-new install with the "fast" preset applied.
def _build_defaults() -> Dict[str, Any]:
    fast = C.PRESETS["fast"]
    return {
        "schema_version": SCHEMA_VERSION,
        # Window
        "window_geometry": C.WINDOW_GEOMETRY,
        # Languages
        "ui_language": C.DEFAULT_UI_LANGUAGE,
        "transcription_language": C.DEFAULTS["transcription_lang"],
        "auto_lang": False,
        # Sidebar selection
        "preset": "fast",
        # Files
        "audio_dir": "",        # empty -> GUI uses C.resolve_default_audio_dir
        "output_dir": "",       # empty -> GUI uses C.resolve_default_output_dir
        # Model
        "model_path": C.DEFAULT_MODEL,
        # Compute
        "device": "cuda",       # GUI overrides to "cpu" when no CUDA
        "dtype": "auto",
        "batch_size": fast["batch_size_cuda"],
        # Decoding
        "decode_profile": fast["decode_profile"],
        "max_new_tokens": fast["max_new_tokens"],
        # Segmentation
        "chunk_sec": fast["chunk_sec"],
        "overlap_sec": fast["overlap_sec"],
        "target_db": fast["target_db"],
        # VAD
        "use_vad": fast["vad_enabled"],
        "vad_threshold": fast["vad_threshold"],
        "vad_silence_ms": fast["vad_silence_ms"],
        "vad_merge_gap": fast["vad_merge_gap"],
        # Output
        "save_srt": False,
        # Appearance (Phase F): "Dark" | "Light" | "System"
        "appearance_mode": C.APPEARANCE_MODE,
    }


DEFAULTS: Dict[str, Any] = _build_defaults()


def load(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load settings, falling back to defaults for missing/invalid file.

    Never raises. Unknown extra keys are kept (forward-compat with newer
    versions of the app that may have added them).
    """
    target = path or get_default_path()
    merged = dict(DEFAULTS)
    if not target.exists():
        return merged
    try:
        with target.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return merged
        # Only accept JSON-serializable scalars + lists. Skip anything weird.
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool, list, type(None))):
                merged[key] = value
    except (OSError, json.JSONDecodeError):
        # Corrupt or unreadable file: act like it doesn't exist.
        return dict(DEFAULTS)
    return merged


def save(settings: Dict[str, Any], path: Optional[Path] = None) -> bool:
    """Atomically write settings to disk. Returns True on success."""
    target = path or get_default_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file in the same directory so os.replace is
        # atomic on Windows (cross-volume rename would not be).
        fd, tmp_path = tempfile.mkstemp(
            prefix=".settings.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
        finally:
            # If replace succeeded the tmp is gone; if it failed, clean up.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return True
    except OSError:
        return False
