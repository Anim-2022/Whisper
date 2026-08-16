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
  (old files don't have it -> default).
- Unknown keys are DROPPED on load. This reverses the previous forward-compat
  policy deliberately: a v1 file carries fields like `chunk_sec` and `dtype`
  that describe an engine that no longer exists, and letting them through would
  feed them to widgets that were removed.
- The GUI is the source of truth for *current* state; settings_store just
  remembers the last-known good snapshot.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from . import constants as C

SETTINGS_DIRNAME = ".whisper_gui"
SETTINGS_FILENAME = "settings.json"
SCHEMA_VERSION = 2

#: Suffix for the one-time backup taken before a v1 file is migrated.
BACKUP_SUFFIX = ".v1.bak.json"


def get_default_path() -> Path:
    """Return absolute path to the settings file under the user's home dir."""
    return Path.home() / SETTINGS_DIRNAME / SETTINGS_FILENAME


# Defaults for any missing key. Matches a brand-new install with the default
# preset applied.
def _build_defaults() -> dict[str, Any]:
    preset = C.PRESETS[C.DEFAULT_PRESET]
    return {
        "schema_version": SCHEMA_VERSION,
        # Window
        "window_geometry": C.WINDOW_GEOMETRY,
        # Languages
        "ui_language": C.DEFAULT_UI_LANGUAGE,
        "transcription_language": C.DEFAULTS["transcription_lang"],
        "auto_lang": False,
        "initial_prompt": "",
        # Sidebar selection
        "preset": C.DEFAULT_PRESET,
        # Files
        "audio_dir": "",        # empty -> GUI uses C.resolve_default_audio_dir
        "output_dir": "",       # empty -> GUI uses C.resolve_default_output_dir
        # Model
        "model_id": C.DEFAULT_MODEL,
        # Compute
        "device": "cuda",       # GUI overrides to "cpu" when no CUDA
        "compute_type": "auto",
        "batched": preset["batched"],
        "batch_size": preset["batch_size_cuda"],
        # Decoding
        "beam_size": preset["beam_size"],
        "temperature_fallback": preset["temperature_fallback"],
        "condition_on_previous_text": preset["condition_on_previous_text"],
        "word_timestamps": preset["word_timestamps"],
        "no_speech_threshold": preset["no_speech_threshold"],
        "compression_ratio_threshold": preset["compression_ratio_threshold"],
        "log_prob_threshold": preset["log_prob_threshold"],
        "hallucination_silence_threshold": preset["hallucination_silence_threshold"],
        # VAD
        "vad_filter": preset["vad_enabled"],
        "vad_threshold": preset["vad_threshold"],
        "vad_min_speech_ms": preset["vad_min_speech_ms"],
        "vad_min_silence_ms": preset["vad_min_silence_ms"],
        "vad_speech_pad_ms": preset["vad_speech_pad_ms"],
        # Output
        "formats": list(C.DEFAULT_OUTPUT_FORMATS),
    }


DEFAULTS: dict[str, Any] = _build_defaults()

#: v1 keys describing the removed transformers pipeline. Listed explicitly so
#: the migration is self-documenting rather than relying on the whitelist alone.
_DEAD_V1_KEYS = frozenset({
    "chunk_sec", "overlap_sec", "target_db", "vad_merge_gap", "max_new_tokens",
    "dtype", "decode_profile", "use_vad", "vad_silence_ms", "save_srt", "model_path",
})

#: Old HF repo ids -> the CT2 directory names produced by tools/convert_models.py.
def _hf_id_to_ct2_name(model_path: str) -> str:
    """"openai/whisper-medium" -> "whisper-medium"; a real path is kept as-is."""
    text = str(model_path or "").strip()
    if not text:
        return C.DEFAULT_MODEL
    if Path(text).is_dir():
        return text
    return text.split("/")[-1] or C.DEFAULT_MODEL


def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Translate a v1 snapshot into the v2 schema.

    bfloat16 has no CTranslate2 equivalent, so it maps to "auto"; the old
    balanced/quality decode profile becomes the batched/sequential mode.
    """
    out = dict(data)
    out["compute_type"] = {
        "auto": "auto", "float16": "float16", "bfloat16": "auto", "float32": "float32",
    }.get(data.get("dtype"), "auto")
    out["batched"] = data.get("decode_profile", "balanced") == "balanced"
    out["model_id"] = _hf_id_to_ct2_name(data.get("model_path", ""))
    out["vad_filter"] = bool(data.get("use_vad", True))
    out["vad_min_silence_ms"] = data.get("vad_silence_ms", DEFAULTS["vad_min_silence_ms"])
    out["formats"] = ["txt"] + (["srt"] if data.get("save_srt") else [])
    for key in _DEAD_V1_KEYS:
        out.pop(key, None)
    return out


#: version -> function producing the next version.
MIGRATIONS = {1: _migrate_v1_to_v2}


def _backup_once(target: Path, version: int) -> None:
    """Keep a copy of a pre-migration file, so a downgrade is still possible."""
    if version >= SCHEMA_VERSION:
        return
    backup = target.with_suffix("")
    backup = backup.with_name(backup.name + BACKUP_SUFFIX)
    if backup.exists():
        return
    try:
        backup.write_bytes(target.read_bytes())
    except OSError:
        pass  # a missing backup must never block startup


def load(path: Path | None = None) -> dict[str, Any]:
    """Load settings, falling back to defaults for a missing/invalid file.

    Never raises: a corrupt file, a non-dict payload, an unparseable version or
    an exception inside a migration all yield defaults. The GUI must always start.
    """
    target = path or get_default_path()
    if not target.exists():
        return dict(DEFAULTS)

    try:
        with target.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULTS)

        try:
            version = int(data.get("schema_version", 1))
        except (TypeError, ValueError):
            version = 1

        _backup_once(target, version)

        while version < SCHEMA_VERSION and version in MIGRATIONS:
            data = MIGRATIONS[version](data)
            version += 1

        merged = dict(DEFAULTS)
        # Whitelist against DEFAULTS: anything we no longer understand is dropped
        # rather than passed through to the GUI.
        for key, value in data.items():
            if key in DEFAULTS and isinstance(value, str | int | float | bool | list | type(None)):
                merged[key] = value
        merged["schema_version"] = SCHEMA_VERSION
        return merged
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return dict(DEFAULTS)


def save(settings: dict[str, Any], path: Path | None = None) -> bool:
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
