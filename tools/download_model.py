"""Download a ready-made CTranslate2 Whisper model into models/ct2/.

This is the simple path for a fresh install: the models on the Hub are already
converted, so nothing here needs torch or transformers, and there is no
conversion step. Use tools/convert_models.py instead only when you already have
HuggingFace weights on disk and want to convert them yourself offline.

    python tools/download_model.py                 # the recommended default
    python tools/download_model.py small large-v3  # specific models
    python tools/download_model.py --list          # what can be downloaded

Downloading is a one-time setup step. Once a model is in models/ct2/ the
application never touches the network again — it loads with local_files_only.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CT2_ROOT = REPO_ROOT / "models" / "ct2"

# Local directory name -> Hub repository holding the converted model.
# Kept explicit rather than imported from faster_whisper so the local names stay
# stable if that mapping ever changes upstream.
MODELS: dict[str, str] = {
    "whisper-tiny": "Systran/faster-whisper-tiny",
    "whisper-base": "Systran/faster-whisper-base",
    "whisper-small": "Systran/faster-whisper-small",
    "whisper-medium": "Systran/faster-whisper-medium",
    "whisper-large-v3": "Systran/faster-whisper-large-v3",
    "whisper-large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

DEFAULT_MODEL = "whisper-medium"

# Number of mel filterbank channels each architecture expects.
#
# The converted repositories on the Hub ship no preprocessor_config.json, and
# without one faster-whisper's feature extractor silently defaults to 80. That is
# correct for everything up to medium and wrong for large-v3 and turbo, which
# were trained on 128 — a model loaded with the wrong count produces confident
# nonsense rather than an error. So the file is written after download, which
# both fixes the trap and satisfies validate_ct2_dir().
MEL_BINS = {
    "whisper-tiny": 80,
    "whisper-base": 80,
    "whisper-small": 80,
    "whisper-medium": 80,
    "whisper-large-v3": 128,
    "whisper-large-v3-turbo": 128,
}


def write_preprocessor_config(target: Path, name: str) -> None:
    """Create preprocessor_config.json when the repository does not ship one."""
    import json

    path = target / "preprocessor_config.json"
    if path.is_file():
        return
    path.write_text(json.dumps({
        "chunk_length": 30,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": MEL_BINS[name],
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "WhisperProcessor",
        "return_attention_mask": False,
        "sampling_rate": 16000,
    }, indent=2), encoding="utf-8")

# Rough download sizes, so the choice is informed before it starts.
SIZES = {
    "whisper-tiny": "75 MB",
    "whisper-base": "145 MB",
    "whisper-small": "480 MB",
    "whisper-medium": "1.5 GB",
    "whisper-large-v3": "3.1 GB",
    "whisper-large-v3-turbo": "1.6 GB",
}

NOTES = {
    "whisper-medium": "recommended default — best balance of speed and accuracy",
    "whisper-small": "faster and lighter, less accurate on difficult speech",
    "whisper-large-v3": "highest quality, heaviest",
    "whisper-large-v3-turbo": "fastest, but loses punctuation on long speech",
    "whisper-tiny": "for smoke-testing the install, not for real transcription",
    "whisper-base": "for smoke-testing the install, not for real transcription",
}


def normalize(name: str) -> str:
    """Accept both "medium" and "whisper-medium"."""
    name = name.strip()
    if name in MODELS:
        return name
    prefixed = f"whisper-{name}"
    if prefixed in MODELS:
        return prefixed
    raise SystemExit(
        f"Unknown model {name!r}.\nAvailable: "
        + ", ".join(sorted(k.replace("whisper-", "") for k in MODELS))
    )


def already_present(target: Path) -> bool:
    return (target / "model.bin").is_file() and (target / "config.json").is_file()


def print_catalogue() -> None:
    print(f"{'model':26}{'size':>9}  notes")
    for name in MODELS:
        marker = " (installed)" if already_present(CT2_ROOT / name) else ""
        print(f"{name:26}{SIZES.get(name, '?'):>9}  {NOTES.get(name, '')}{marker}")


def download(name: str, force: bool) -> bool:
    from huggingface_hub import snapshot_download

    target = CT2_ROOT / name
    if already_present(target) and not force:
        print(f"{name}: already present, skipping (use --force to redownload)")
        return True

    repo = MODELS[name]
    print(f"{name}: downloading {repo} (~{SIZES.get(name, '?')})")
    try:
        # allow_patterns keeps the .bin/.json files and skips the repo's README
        # and any PyTorch duplicates some of these repos also carry.
        snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            allow_patterns=["*.bin", "*.json", "*.txt"],
        )
    except Exception as exc:  # noqa: BLE001 - report and continue with the rest
        print(f"{name}: FAILED — {type(exc).__name__}: {exc}")
        return False

    write_preprocessor_config(target, name)

    missing = [f for f in ("model.bin", "config.json", "tokenizer.json",
                           "preprocessor_config.json") if not (target / f).is_file()]
    if not (target / "vocabulary.json").is_file() and not (target / "vocabulary.txt").is_file():
        missing.append("vocabulary.json|vocabulary.txt")
    if missing:
        print(f"{name}: incomplete download, missing {', '.join(missing)}")
        return False

    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024 ** 3
    print(f"{name}: ready ({size:.1f} GB) -> {target}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*",
                        help=f"model names (default: {DEFAULT_MODEL})")
    parser.add_argument("--list", action="store_true", help="show what can be downloaded")
    parser.add_argument("--force", action="store_true", help="redownload if already present")
    args = parser.parse_args(argv)

    if args.list:
        print_catalogue()
        return 0

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is missing. Install the runtime first:\n"
              "    pip install -r requirements.txt", file=sys.stderr)
        return 2

    names = [normalize(n) for n in (args.models or [DEFAULT_MODEL])]
    CT2_ROOT.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in names:
        if not download(name, args.force):
            failures += 1

    if not failures:
        free = shutil.disk_usage(CT2_ROOT).free / 1024 ** 3
        print(f"\nDone. Launch with Start_Whisper.bat  ({free:.0f} GB free on this drive)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
