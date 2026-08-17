"""Grid row allocation.

Hand-numbered grid rows fail silently: the field helpers draw their help line on
`row + 1`, so inserting one field pushes a section's last help line onto the next
section's heading and the two labels are drawn on top of each other. RowCounter
makes that arithmetic impossible to get wrong.

The RowCounter tests below are pure and always run. The collision audit needs a
real Tk display, so it is marked `gui` and excluded by default — run it with
`pytest -m gui` on a desktop machine.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _row_counter():
    """Extract RowCounter without importing field_factory, which needs Tk."""
    source = (REPO_ROOT / "gui" / "widgets" / "field_factory.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == "RowCounter":
            namespace: dict = {}
            exec(compile(ast.Module([node], []), "<field_factory>", "exec"), namespace)
            return namespace["RowCounter"]
    raise AssertionError("RowCounter not found in field_factory.py")


@pytest.fixture(scope="module")
def counter_cls():
    return _row_counter()


def test_rows_are_handed_out_in_order(counter_cls):
    rows = counter_cls()
    assert [rows.take(), rows.take(), rows.take()] == [0, 1, 2]


def test_multi_row_widgets_reserve_their_help_line(counter_cls):
    rows = counter_cls()
    assert rows.take(2) == 0      # field on 0, help on 1
    assert rows.take() == 2       # next widget must not land on the help line


def test_no_row_is_ever_handed_out_twice(counter_cls):
    rows = counter_cls()
    handed_out: list[int] = []
    for count in (1, 2, 2, 1, 2, 1, 2, 2):
        start = rows.take(count)
        handed_out.extend(range(start, start + count))
    assert len(handed_out) == len(set(handed_out))
    assert handed_out == sorted(handed_out)


def test_start_offset_and_next_row(counter_cls):
    rows = counter_cls(start=5)
    assert rows.take() == 5
    assert rows.next_row == 6


@pytest.mark.parametrize("bad", [0, -1])
def test_zero_or_negative_is_rejected(counter_cls, bad):
    with pytest.raises(ValueError):
        counter_cls().take(bad)


# --- the invariant itself, on a real window --------------------------------
@pytest.mark.gui
def test_no_widgets_overlap_in_any_tab():
    """Two widgets sharing a grid cell in the same frame is the actual bug.

    This is what caught the VAD padding hint being drawn on top of the
    "Decoding" heading, and it checks the property directly rather than
    inspecting the source for hardcoded numbers.
    """
    from collections import defaultdict

    import whisper_gui

    app = whisper_gui.WhisperGUI()
    try:
        # The Logs tab is a single textbox packed rather than gridded, so only
        # the two form-style tabs can have cell collisions.
        for name, frame in (("Settings", app.settings_frame),
                            ("Advanced", app.adv_frame)):
            occupants = defaultdict(list)
            for widget in frame.winfo_children():
                info = widget.grid_info()
                if not info:
                    continue
                row, col = int(info["row"]), int(info["column"])
                for r in range(row, row + int(info.get("rowspan", 1))):
                    for c in range(col, col + int(info.get("columnspan", 1))):
                        occupants[(r, c)].append(type(widget).__name__)
            clashes = {cell: names for cell, names in occupants.items() if len(names) > 1}
            assert not clashes, f"{name} tab has overlapping widgets: {clashes}"
    finally:
        app.destroy()
