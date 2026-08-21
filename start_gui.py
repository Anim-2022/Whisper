import sys
from pathlib import Path

# Minimum supported interpreter. datetime.UTC (whisper_engine/engine.py) landed in
# 3.11, and several runtime `X | Y` isinstance checks need 3.10. Check before the
# first import so an unsupported interpreter gets a clear message rather than an
# ImportError from somewhere deep in the package.
MIN_PYTHON = (3, 11)

if sys.version_info < MIN_PYTHON:
    sys.exit(
        f"Whisper Unified needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer, "
        f"but this is {sys.version.split()[0]}.\n"
        f"Create the virtual environment with a newer interpreter:\n"
        f"    py -3.12 -m venv .venv"
    )

# Ensure local directory is in path
sys.path.append(str(Path(__file__).parent))

from whisper_gui import WhisperGUI  # noqa: E402  (must follow the version check)

if __name__ == "__main__":
    app = WhisperGUI()
    app.mainloop()
