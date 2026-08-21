"""Offline speech-to-text engine built on faster-whisper (CTranslate2).

Import order note: :mod:`whisper_engine.cuda_dlls` must run before anything
imports ``ctranslate2``. :mod:`whisper_engine.engine` handles that; nothing else
in this package imports faster_whisper at module scope, which keeps the pure
modules (types, timestamps, writers, progress, config, models) importable —
and testable — without the native runtime installed.
"""

__all__ = ["__version__"]

__version__ = "2.0.0-dev"
