import pytest

from whisper_engine.models import (
    ModelIncompleteError,
    ModelNotFoundError,
    discover_ct2_models,
    is_ct2_model_dir,
    missing_files,
    resolve_ct2_model_dir,
    validate_ct2_dir,
)


def test_discovery_lists_only_real_models(fake_ct2_dir, tmp_path):
    root = tmp_path / "ct2"
    fake_ct2_dir("whisper-medium", root=root)
    fake_ct2_dir("whisper-small", root=root)
    (root / "not-a-model").mkdir()
    (root / "README.txt").write_text("x", encoding="utf-8")
    assert discover_ct2_models(root) == ["whisper-medium", "whisper-small"]


def test_discovery_on_missing_root_is_empty(tmp_path):
    assert discover_ct2_models(tmp_path / "nope") == []


def test_is_ct2_model_dir_requires_weights(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    assert not is_ct2_model_dir(d)
    (d / "model.bin").write_text("x", encoding="utf-8")
    assert is_ct2_model_dir(d)


def test_resolve_by_name(fake_ct2_dir, tmp_path):
    root = tmp_path / "ct2"
    expected = fake_ct2_dir("whisper-medium", root=root)
    assert resolve_ct2_model_dir("whisper-medium", root=root) == expected


def test_resolve_by_explicit_path(fake_ct2_dir, tmp_path):
    d = fake_ct2_dir("elsewhere", root=tmp_path / "custom")
    assert resolve_ct2_model_dir(str(d)) == d


def test_unknown_model_lists_the_alternatives(fake_ct2_dir, tmp_path):
    root = tmp_path / "ct2"
    fake_ct2_dir("whisper-small", root=root)
    with pytest.raises(ModelNotFoundError) as exc:
        resolve_ct2_model_dir("whisper-large", root=root)
    assert "whisper-small" in str(exc.value)
    assert "convert_models.py" in str(exc.value)


def test_empty_model_id_rejected():
    with pytest.raises(ModelNotFoundError):
        resolve_ct2_model_dir("")


@pytest.mark.parametrize("omit", ["tokenizer.json", "preprocessor_config.json",
                                  "config.json", "vocabulary.json"])
def test_missing_required_file_is_reported_by_name(fake_ct2_dir, tmp_path, omit):
    """Both of these fail silently at runtime, so they must fail loudly at load.

    Without tokenizer.json faster-whisper reaches the network; without
    preprocessor_config.json a 128-mel model decodes as 80 and emits garbage.
    """
    d = fake_ct2_dir("m", root=tmp_path / "ct2", omit=(omit,))
    assert omit.split(".")[0] in " ".join(missing_files(d))
    with pytest.raises(ModelIncompleteError) as exc:
        validate_ct2_dir(d, "m")
    assert "convert_models.py" in str(exc.value)


def test_complete_model_validates(fake_ct2_dir, tmp_path):
    d = fake_ct2_dir("m", root=tmp_path / "ct2")
    assert missing_files(d) == []
    validate_ct2_dir(d, "m")


def test_vocabulary_txt_is_accepted(fake_ct2_dir, tmp_path):
    """Older CTranslate2 versions write vocabulary.txt instead of .json."""
    d = fake_ct2_dir("m", root=tmp_path / "ct2", omit=("vocabulary.json",))
    (d / "vocabulary.txt").write_text("x", encoding="utf-8")
    assert missing_files(d) == []


def test_validate_on_missing_directory(tmp_path):
    with pytest.raises(ModelNotFoundError):
        validate_ct2_dir(tmp_path / "nope", "m")
