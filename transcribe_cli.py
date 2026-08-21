"""Headless transcription runner.

Useful for batch jobs, for A/B comparisons against evaluate_transcriptions.py,
and as a smoke test of the engine without launching the GUI.

    python transcribe_cli.py audio/                     --formats txt,srt
    python transcribe_cli.py audio/meeting.m4a          --model whisper-medium
    python transcribe_cli.py audio/ --preset difficult  --out predictions/new
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from gui import constants as C
from whisper_core import WhisperCore
from whisper_engine.audio import find_audio_files
from whisper_engine.config import TranscriptionConfig
from whisper_engine.device import has_cuda
from whisper_engine.models import discover_ct2_models
from whisper_engine.progress import parse_status
from whisper_engine.timestamps import format_duration


def collect_inputs(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            files.extend(find_audio_files(path))
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: {target} not found, skipping", file=sys.stderr)
    # Preserve order but drop duplicates when a file is named twice.
    return list(dict.fromkeys(files))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", default=["audio"],
                   help="audio files and/or directories (default: audio/)")
    p.add_argument("--model", default=None, help=f"default: {C.DEFAULT_MODEL}")
    p.add_argument("--preset", default=C.DEFAULT_PRESET, choices=list(C.PRESETS),
                   help=f"default: {C.DEFAULT_PRESET}")
    p.add_argument("--lang", default="ru")
    p.add_argument("--auto-lang", action="store_true")
    p.add_argument("--formats", default="txt",
                   help="comma-separated: txt,srt,vtt,json,md")
    p.add_argument("--out", type=Path, default=Path("audio_to_text"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--compute-type", default="auto")
    p.add_argument("--initial-prompt", default=None,
                   help="seed vocabulary: attendee names, product jargon")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--quiet", action="store_true", help="only print per-file results")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        models = discover_ct2_models()
        print("\n".join(models) if models else
              "No converted models. Run: python tools/convert_models.py")
        return 0

    files = collect_inputs(args.inputs)
    if not files:
        print("No audio files found.", file=sys.stderr)
        return 1

    config = TranscriptionConfig(
        model_id=args.model or C.DEFAULT_MODEL,
        device=args.device,
        compute_type=args.compute_type,
        lang=args.lang,
        auto_lang=args.auto_lang,
        initial_prompt=args.initial_prompt,
        formats=tuple(f.strip() for f in args.formats.split(",") if f.strip()),
        output_dir=args.out,
        **C.preset_to_config_kwargs(C.PRESETS[args.preset], has_cuda=has_cuda()),
    )

    print(f"{len(files)} file(s) -> {config.output_dir}")
    print(f"preset={args.preset}  {config.describe()}")

    # Problems are printed even with --quiet: a run that fails silently and still
    # reports a total looks like success, which is how someone concludes the tool
    # is broken and gives up.
    problem = re.compile(r"error|failed|fail|falling back|no speech|missing",
                         re.IGNORECASE)

    def on_log(message: str) -> None:
        if not args.quiet or problem.search(message):
            print(message, flush=True)

    def on_progress(_fraction: float, status: str) -> None:
        parsed = parse_status(status)
        if parsed.kind == "done_file":
            print(f"  done: {parsed.name}", flush=True)

    core = WhisperCore(on_log=on_log, on_progress=on_progress)

    started = time.monotonic()
    try:
        core.process_files(files, config)
    except KeyboardInterrupt:
        core.request_stop()
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print(f"Total: {format_duration(time.monotonic() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
