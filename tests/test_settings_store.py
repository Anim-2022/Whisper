"""Settings persistence, including the v1 -> v2 migration.

The migration is the risky part: a v1 file carries keys describing the removed
transformers pipeline, and the previous load() merged *every* key it found. Left
alone, those would be handed to widgets that no longer exist.
"""
from __future__ import annotations

import json

import pytest

from gui import settings_store as store


@pytest.fixture
def settings_path(tmp_path):
    return tmp_path / ".whisper_gui" / "settings.json"


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


V1_FILE = {
    "schema_version": 1,
    "window_geometry": "1200x900",
    "ui_language": "en",
    "transcription_language": "de",
    "preset": "noisy",
    "model_path": "openai/whisper-medium",
    "device": "cuda",
    "dtype": "bfloat16",
    "batch_size": 4,
    "decode_profile": "quality",
    "max_new_tokens": 160,
    "chunk_sec": 20.0,
    "overlap_sec": 3.0,
    "target_db": -20.0,
    "use_vad": True,
    "vad_threshold": 0.45,
    "vad_silence_ms": 140,
    "vad_merge_gap": 0.25,
    "save_srt": True,
}


# --- basics ----------------------------------------------------------------
def test_missing_file_returns_defaults(settings_path):
    assert store.load(settings_path) == store.DEFAULTS


def test_round_trip(settings_path):
    snapshot = dict(store.DEFAULTS, ui_language="en", batch_size=8)
    assert store.save(snapshot, settings_path)
    assert store.load(settings_path)["batch_size"] == 8


def test_save_leaves_no_temp_file(settings_path):
    store.save(dict(store.DEFAULTS), settings_path)
    assert [p.name for p in settings_path.parent.iterdir()] == [settings_path.name]


@pytest.mark.parametrize("payload", ["{ not json", '"a string"', "[1,2,3]", "null"])
def test_malformed_files_yield_defaults(settings_path, payload):
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(payload, encoding="utf-8")
    assert store.load(settings_path) == store.DEFAULTS


def test_unparseable_version_is_treated_as_v1(settings_path):
    write(settings_path, dict(V1_FILE, schema_version="banana"))
    assert store.load(settings_path)["schema_version"] == store.SCHEMA_VERSION


# --- the whitelist ---------------------------------------------------------
def test_unknown_keys_are_dropped(settings_path):
    write(settings_path, {"schema_version": 2, "totally_made_up": 1, "batch_size": 7})
    loaded = store.load(settings_path)
    assert "totally_made_up" not in loaded
    assert loaded["batch_size"] == 7


# --- migration -------------------------------------------------------------
def test_v1_dead_keys_do_not_survive(settings_path):
    write(settings_path, V1_FILE)
    loaded = store.load(settings_path)
    for dead in store._DEAD_V1_KEYS:
        assert dead not in loaded, f"{dead} leaked through the migration"


def test_v1_values_are_translated(settings_path):
    write(settings_path, V1_FILE)
    loaded = store.load(settings_path)
    assert loaded["model_id"] == "whisper-medium"
    # bfloat16 has no CTranslate2 equivalent.
    assert loaded["compute_type"] == "auto"
    # "quality" was the sequential-style profile.
    assert loaded["batched"] is False
    assert loaded["formats"] == ["txt", "srt"]
    assert loaded["vad_filter"] is True
    assert loaded["vad_min_silence_ms"] == 140


def test_v1_without_srt_keeps_only_txt(settings_path):
    write(settings_path, dict(V1_FILE, save_srt=False))
    assert store.load(settings_path)["formats"] == ["txt"]


def test_v1_balanced_profile_becomes_batched(settings_path):
    write(settings_path, dict(V1_FILE, decode_profile="balanced"))
    assert store.load(settings_path)["batched"] is True


def test_untouched_v1_keys_are_preserved(settings_path):
    write(settings_path, V1_FILE)
    loaded = store.load(settings_path)
    assert loaded["ui_language"] == "en"
    assert loaded["transcription_language"] == "de"
    assert loaded["preset"] == "noisy"
    assert loaded["window_geometry"] == "1200x900"


def test_migration_bumps_the_version(settings_path):
    write(settings_path, V1_FILE)
    assert store.load(settings_path)["schema_version"] == store.SCHEMA_VERSION


def test_v1_file_is_backed_up_once(settings_path):
    write(settings_path, V1_FILE)
    store.load(settings_path)
    backups = [p for p in settings_path.parent.iterdir() if store.BACKUP_SUFFIX in p.name]
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8"))["dtype"] == "bfloat16"

    # A second load must not overwrite the original backup.
    store.save(dict(store.DEFAULTS), settings_path)
    store.load(settings_path)
    assert json.loads(backups[0].read_text(encoding="utf-8"))["dtype"] == "bfloat16"


def test_v2_file_is_not_backed_up(settings_path):
    store.save(dict(store.DEFAULTS), settings_path)
    store.load(settings_path)
    assert not [p for p in settings_path.parent.iterdir() if store.BACKUP_SUFFIX in p.name]


def test_a_failing_migration_falls_back_to_defaults(settings_path, monkeypatch):
    def boom(_data):
        raise ValueError("bad data")

    monkeypatch.setitem(store.MIGRATIONS, 1, boom)
    write(settings_path, V1_FILE)
    assert store.load(settings_path) == store.DEFAULTS


# --- defaults integrity ----------------------------------------------------
def test_defaults_cover_every_key_the_gui_reads():
    """_collect_settings_snapshot indexes DEFAULTS directly, so a gap is a KeyError."""
    required = {
        "model_id", "device", "compute_type", "batched", "batch_size", "beam_size",
        "temperature_fallback", "condition_on_previous_text", "word_timestamps",
        "no_speech_threshold", "compression_ratio_threshold", "log_prob_threshold",
        "hallucination_silence_threshold", "vad_filter", "vad_threshold",
        "vad_min_speech_ms", "vad_min_silence_ms", "vad_speech_pad_ms",
        "formats", "initial_prompt", "transcription_language",
    }
    assert required <= set(store.DEFAULTS)


def test_defaults_are_json_serializable():
    json.dumps(store.DEFAULTS)
