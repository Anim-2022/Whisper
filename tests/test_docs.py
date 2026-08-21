"""The READMEs must not drift out of step with the code.

Every problem asserted here was found by hand after it had already shipped: the
preset lineup was renamed while the feature list and the preset table still
advertised the old names, the in-app hints told users to pick a preset that no
longer existed, and all three READMEs linked to a file that is gitignored and
therefore absent on GitHub.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from gui import constants as C
from gui.i18n import SUPPORTED_LANGUAGES, UI_TEXT

REPO_ROOT = Path(__file__).resolve().parent.parent
READMES = ("README.md", "README.ru.md", "README.uk.md")

#: Names of the presets that existed before the benchmark-driven rewrite.
RETIRED_PRESET_NAMES = ("Noisy audio", "Шумное аудио", "Шумне аудіо")


@pytest.fixture(scope="module")
def tracked_files() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=False)
    return set(out.stdout.split())


@pytest.mark.parametrize("name", READMES)
def test_relative_links_point_at_files_that_are_in_the_repository(name, tracked_files):
    """A link to a gitignored file is a 404 on GitHub even though it resolves locally."""
    if not tracked_files:
        pytest.skip("git not available")
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    broken = []
    for link in re.findall(r"\]\(([^)#][^)]*)\)", text):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target = link.split("#")[0]
        if target and target not in tracked_files:
            broken.append(link)
    assert not broken, f"{name} links to files absent from the repo: {broken}"


@pytest.mark.parametrize("name", READMES)
def test_readmes_do_not_advertise_retired_presets(name):
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    found = [n for n in RETIRED_PRESET_NAMES if n in text]
    assert not found, f"{name} still mentions retired presets: {found}"


@pytest.mark.parametrize("name", READMES)
def test_readmes_name_the_current_presets(name):
    """Each README must list every preset, in whichever language it is written."""
    text = (REPO_ROOT / name).read_text(encoding="utf-8").lower()
    expected = {
        "README.md": ("meeting", "long recording", "difficult audio"),
        "README.ru.md": ("встреча", "длинная запись", "сложное аудио"),
        "README.uk.md": ("зустріч", "довгий запис", "складне аудіо"),
    }[name]
    missing = [w for w in expected if w not in text]
    assert not missing, f"{name} does not mention: {missing}"


def test_readmes_stay_structurally_in_step():
    """The translations are line-for-line equivalents; a section added to one
    without the others is how they start diverging."""
    counts = {n: (REPO_ROOT / n).read_text(encoding="utf-8").count("\n## ") for n in READMES}
    assert len(set(counts.values())) == 1, f"section counts differ: {counts}"


@pytest.mark.parametrize("lang", list(SUPPORTED_LANGUAGES))
def test_ui_hints_do_not_recommend_a_preset_that_does_not_exist(lang):
    """The sidebar told people to choose "Fast" long after it was removed."""
    live = {UI_TEXT[lang][f"preset_{k}"].lower() for k in C.PRESET_KEYS}
    for key in ("sidebar_hint", "settings_intro_body"):
        text = UI_TEXT[lang][key].lower()
        for retired in ("fast", "accurate", "быстрый профиль", "точный профиль"):
            if retired in live:
                continue
            assert retired not in text, f"{lang}/{key} mentions retired preset {retired!r}"


def test_documented_default_model_matches_the_code():
    for name in READMES:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert C.DEFAULT_MODEL in text, f"{name} never names the default model"
