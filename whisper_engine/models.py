"""Discovery and validation of local CTranslate2 Whisper models.

Everything is local-only by design. Two files decide whether a converted model
works at all, and both fail silently rather than loudly, so :func:`validate_ct2_dir`
checks them explicitly:

``tokenizer.json``
    Missing, faster-whisper falls back to ``Tokenizer.from_pretrained(
    "openai/whisper-tiny")`` — a network call that breaks the offline guarantee.

``preprocessor_config.json``
    Missing, the feature extractor defaults to 80 mel bins. large-v3 and
    large-v3-turbo use 128, so the model would quietly emit garbage.
"""
from __future__ import annotations

from pathlib import Path

CT2_ROOT_SUBDIR = "ct2"
MODEL_WEIGHTS_FILE = "model.bin"
REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json")


class ModelNotFoundError(FileNotFoundError):
    """Raised when a requested model has no usable directory on disk."""


class ModelIncompleteError(FileNotFoundError):
    """Raised when a directory looks like a model but is missing required files."""


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def models_root() -> Path:
    return repo_root() / "models"


def ct2_root() -> Path:
    return models_root() / CT2_ROOT_SUBDIR


def _has_vocabulary(path: Path) -> bool:
    # CTranslate2 writes vocabulary.json on current versions, vocabulary.txt on older ones.
    return (path / "vocabulary.json").is_file() or (path / "vocabulary.txt").is_file()


def missing_files(path: Path) -> list[str]:
    """Required files absent from ``path``. Empty list means the model is usable."""
    missing = [f for f in REQUIRED_FILES if not (path / f).is_file()]
    if not _has_vocabulary(path):
        missing.append("vocabulary.json|vocabulary.txt")
    return missing


def is_ct2_model_dir(path: Path) -> bool:
    """Cheap check used by discovery: does this look like a converted model?"""
    return path.is_dir() and (path / MODEL_WEIGHTS_FILE).is_file()


def discover_ct2_models(root: Path | None = None) -> list[str]:
    """Names of converted models under ``models/ct2``, sorted for a stable combo box."""
    base = root or ct2_root()
    if not base.is_dir():
        return []
    return sorted(d.name for d in base.iterdir() if is_ct2_model_dir(d))


def conversion_hint(model_id: str) -> str:
    """Both ways to obtain a model, download first.

    Pointing only at the converter is a dead end on a fresh install: it has
    nothing to convert and says so, which leaves the user going in circles.
    """
    short = model_id.replace("whisper-", "") or "medium"
    return (
        "Get it with either:\n"
        f"    python tools/download_model.py {short}"
        "        (a ready-made model, no torch needed)\n"
        f"    python tools/convert_models.py --only {model_id}"
        "   (converts HuggingFace weights you already have)"
    )


def validate_ct2_dir(path: Path, model_id: str = "") -> None:
    """Raise with an actionable message unless ``path`` is a complete CT2 model."""
    name = model_id or path.name
    if not path.is_dir():
        raise ModelNotFoundError(f"Model directory not found: {path}\n{conversion_hint(name)}")
    missing = missing_files(path)
    if missing:
        raise ModelIncompleteError(
            f"Model at {path} is missing: {', '.join(missing)}.\n"
            f"Loading it would either reach the network or decode with the wrong "
            f"number of mel bins.\n{conversion_hint(name)}"
        )


def resolve_ct2_model_dir(model_id: str, root: Path | None = None) -> Path:
    """Resolve a model name or path to a validated CT2 directory.

    Accepts either an explicit path (so a user can point at a model kept
    elsewhere) or a name under ``models/ct2``.
    """
    if not model_id:
        raise ModelNotFoundError("No model selected.")

    candidate = Path(model_id)
    if candidate.is_dir():
        validate_ct2_dir(candidate, model_id)
        return candidate

    base = root or ct2_root()
    target = base / model_id
    if not target.is_dir():
        available = discover_ct2_models(base)
        listing = ", ".join(available) if available else "none found"
        raise ModelNotFoundError(
            f"Model {model_id!r} not found in {base}.\nAvailable: {listing}\n"
            f"{conversion_hint(model_id)}"
        )
    validate_ct2_dir(target, model_id)
    return target
