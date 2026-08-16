"""The two UI languages must stay in lockstep.

A key present in one language and missing in the other raises KeyError the moment
the user switches languages — a crash that is invisible until someone toggles.
"""
from __future__ import annotations

import re

import pytest

from gui.i18n import SUPPORTED_LANGUAGES, UI_TEXT

# Format placeholders each key is expected to carry, so a translation cannot
# quietly drop one and blow up with KeyError at .format() time.
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def test_all_supported_languages_are_present():
    assert set(UI_TEXT) == set(SUPPORTED_LANGUAGES)


def test_languages_have_identical_key_sets():
    reference = set(UI_TEXT["en"])
    for lang in SUPPORTED_LANGUAGES:
        missing = reference - set(UI_TEXT[lang])
        extra = set(UI_TEXT[lang]) - reference
        assert not missing, f"{lang} is missing: {sorted(missing)}"
        assert not extra, f"{lang} has keys 'en' lacks: {sorted(extra)}"


@pytest.mark.parametrize("lang", list(SUPPORTED_LANGUAGES))
def test_no_empty_strings(lang):
    empty = [k for k, v in UI_TEXT[lang].items() if not str(v).strip()]
    assert not empty, f"{lang} has empty values: {empty}"


def test_placeholders_match_across_languages():
    for key, en_value in UI_TEXT["en"].items():
        expected = set(_PLACEHOLDER_RE.findall(str(en_value)))
        for lang in SUPPORTED_LANGUAGES:
            actual = set(_PLACEHOLDER_RE.findall(str(UI_TEXT[lang][key])))
            assert actual == expected, f"{lang}/{key}: {actual} != {expected}"


@pytest.mark.parametrize("lang", list(SUPPORTED_LANGUAGES))
def test_every_string_formats_without_error(lang):
    """Catches stray braces as well as placeholder typos."""
    for key, value in UI_TEXT[lang].items():
        names = _PLACEHOLDER_RE.findall(str(value))
        try:
            str(value).format(**dict.fromkeys(names, "x"))
        except (KeyError, IndexError, ValueError) as exc:
            pytest.fail(f"{lang}/{key} does not format: {exc}")


def test_keys_the_gui_looks_up_dynamically_exist():
    """These are built by string concatenation, so a rename fails silently."""
    from gui import constants as C

    required = (
        [f"mode_{key}" for key in C.MODE_KEYS]
        + [f"checkbox_format_{fmt}" for fmt in C.OUTPUT_FORMATS]
        + [f"preset_{key}" for key in C.PRESET_KEYS]
        + ["status_stage_model", "status_stage_decode",
           "status_stage_vad", "status_stage_lang"]
    )
    for lang in SUPPORTED_LANGUAGES:
        missing = [key for key in required if key not in UI_TEXT[lang]]
        assert not missing, f"{lang} is missing dynamic keys: {missing}"


def test_no_key_still_describes_the_removed_engine():
    """Guards against stale help text surviving a migration."""
    stale = ("decode_profile", "target_db", "merge_gap", "chunk_sec",
             "overlap_sec", "max_new_tokens", "vad_hint", "runtime_vad")
    for lang in SUPPORTED_LANGUAGES:
        leftovers = [k for k in UI_TEXT[lang] if any(s in k for s in stale)]
        assert not leftovers, f"{lang} still defines: {leftovers}"
