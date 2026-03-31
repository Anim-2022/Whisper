import sys
from pathlib import Path

# Ensure local directory is in path
sys.path.append(str(Path(__file__).parent))

from whisper_gui import WhisperGUI

if __name__ == "__main__":
    app = WhisperGUI()
    app.mainloop()
