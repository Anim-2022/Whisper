"""One-time conversion of local HuggingFace Whisper snapshots to CTranslate2.

Runs fully offline against the weights already in ``models/``. Not part of the
application's import graph: this is the only module that needs torch and
transformers (see requirements-convert.txt), so the runtime venv stays slim.

    python tools/convert_models.py                 # convert everything found
    python tools/convert_models.py --only medium   # substring filter
    python tools/convert_models.py --list          # show what would be done

Why float16 and not int8: CTranslate2 re-quantizes on load, so a float16 model
can still be served with compute_type="int8_float16". The reverse is not true.
One conversion keeps every runtime option open.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "models"
CT2_ROOT = MODELS_ROOT / "ct2"

# Files that must be copied verbatim next to the converted model.
#
#   tokenizer.json          - without it faster_whisper falls back to
#                             Tokenizer.from_pretrained("openai/whisper-tiny"),
#                             which is a NETWORK CALL and silently breaks offline use.
#   preprocessor_config.json - without it the feature extractor defaults to 80 mel
#                             bins; large-v3/turbo were trained on 128 and would
#                             silently produce garbage.
COPY_FILES = ("tokenizer.json", "preprocessor_config.json")

# What a usable CT2 Whisper directory must contain after conversion.
REQUIRED_OUTPUT_FILES = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json")


class Candidate(NamedTuple):
    name: str          # "whisper-medium"
    snapshot: Path     # models/models--openai--whisper-medium/snapshots/<ref>
    out_dir: Path      # models/ct2/whisper-medium


def resolve_snapshot(cache_dir: Path) -> Path | None:
    """Pick the snapshot directory of an HF cache entry.

    Prefers refs/main when present. whisper-large-v3-turbo here has a hand-made
    snapshots/main with no refs/ at all, so fall back to the newest snapshot that
    actually holds a config.json.
    """
    snapshots = cache_dir / "snapshots"
    if not snapshots.is_dir():
        return None

    ref_main = cache_dir / "refs" / "main"
    if ref_main.is_file():
        ref = ref_main.read_text(encoding="utf-8").strip()
        candidate = snapshots / ref
        if (candidate / "config.json").is_file():
            return candidate

    usable = [d for d in snapshots.iterdir() if (d / "config.json").is_file()]
    if not usable:
        return None
    return max(usable, key=lambda d: d.stat().st_mtime)


def discover_candidates(models_root: Path, ct2_root: Path) -> list[Candidate]:
    """Find every HF Whisper cache dir under models_root."""
    found: list[Candidate] = []
    for cache_dir in sorted(models_root.glob("models--*")):
        if not cache_dir.is_dir():
            continue
        # models--openai--whisper-medium -> whisper-medium
        name = cache_dir.name.replace("models--", "", 1).split("--")[-1]
        snapshot = resolve_snapshot(cache_dir)
        if snapshot is None:
            print(f"  skip {cache_dir.name}: no snapshot with config.json")
            continue
        found.append(Candidate(name=name, snapshot=snapshot, out_dir=ct2_root / name))
    return found


def verify_output(out_dir: Path) -> list[str]:
    """Return the list of required files missing from a converted directory."""
    missing = [f for f in REQUIRED_OUTPUT_FILES if not (out_dir / f).is_file()]
    # CT2 writes either vocabulary.json or vocabulary.txt depending on version.
    if not (out_dir / "vocabulary.json").is_file() and not (out_dir / "vocabulary.txt").is_file():
        missing.append("vocabulary.json|vocabulary.txt")
    return missing


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 ** 2


def convert_one(cand: Candidate, quantization: str, force: bool) -> bool:
    from ctranslate2.converters import TransformersConverter

    copy_files = [f for f in COPY_FILES if (cand.snapshot / f).is_file()]
    for f in COPY_FILES:
        if f not in copy_files:
            print(f"  ERROR: {cand.snapshot / f} is missing; refusing to convert {cand.name}.")
            print(f"         Converting without {f} produces a model that fails silently.")
            return False

    print(f"  source: {cand.snapshot}")
    print(f"  target: {cand.out_dir}  (quantization={quantization})")

    converter = TransformersConverter(
        str(cand.snapshot),
        copy_files=copy_files,
        load_as_float16=True,
        low_cpu_mem_usage=True,
    )
    converter.convert(str(cand.out_dir), quantization=quantization, force=force)

    missing = verify_output(cand.out_dir)
    if missing:
        print(f"  FAILED verification, missing: {', '.join(missing)}")
        return False

    print(f"  ok — {dir_size_mb(cand.out_dir):.0f} MB")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT,
                        help="directory holding models--*/ cache entries")
    parser.add_argument("--out", type=Path, default=None,
                        help="output root (default: <models-root>/ct2)")
    parser.add_argument("--only", default=None,
                        help="convert only models whose name contains this substring")
    parser.add_argument("--quantization", default="float16",
                        choices=["float32", "float16", "bfloat16", "int16",
                                 "int8", "int8_float32", "int8_float16", "int8_bfloat16"],
                        help="storage precision on disk (default: float16)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output directory")
    parser.add_argument("--list", action="store_true",
                        help="show what would be converted and exit")
    args = parser.parse_args(argv)

    models_root: Path = args.models_root
    ct2_root: Path = args.out or (models_root / "ct2")

    if not models_root.is_dir():
        print(f"No HuggingFace weights found at {models_root}")
        print()
        print("This tool converts weights you already have on disk. If you just")
        print("want a model to use, download a ready-made one instead — no torch")
        print("and no conversion needed:")
        print("    python tools/download_model.py")
        return 2

    print(f"Scanning {models_root}")
    candidates = discover_candidates(models_root, ct2_root)
    if args.only:
        candidates = [c for c in candidates if args.only in c.name]

    if not candidates:
        print("Nothing to convert — no models--*/snapshots/ directories found.")
        print("To download a ready-made model instead:")
        print("    python tools/download_model.py")
        return 1

    if args.list:
        for c in candidates:
            state = "converted" if not verify_output(c.out_dir) else "pending"
            print(f"  {c.name:28} {state:10} <- {c.snapshot}")
        return 0

    ct2_root.mkdir(parents=True, exist_ok=True)

    failures = 0
    for cand in candidates:
        print(f"\n[{cand.name}]")
        if not args.force and not verify_output(cand.out_dir):
            print("  already converted, skipping (use --force to redo)")
            continue
        try:
            if not convert_one(cand, args.quantization, force=True):
                failures += 1
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    print(f"\nDone. {len(candidates) - failures}/{len(candidates)} converted into {ct2_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
