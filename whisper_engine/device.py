"""Compute-device probing without importing torch.

The GUI needs to know whether CUDA is usable before any model is loaded (to grey
out choices and to clamp a persisted ``cuda`` setting on a machine that no longer
has a GPU). Previously this cost a full ``import torch``; here it is a CTranslate2
call behind a try/except, so the GUI module stays importable in CI where neither
CUDA nor CTranslate2 exists.
"""
from __future__ import annotations

#: Values the GUI offers. CTranslate2 has no MPS backend, so "mps" is gone.
DEVICES = ("cuda", "cpu")

#: CTranslate2 compute types we expose. "auto" lets CTranslate2 pick the fastest
#: type the device actually supports.
COMPUTE_TYPES = ("auto", "float16", "int8_float16", "int8", "float32")

_cuda_count: int | None = None


def cuda_device_count(force: bool = False) -> int:
    """Number of usable CUDA devices; 0 if CUDA or CTranslate2 is unavailable."""
    global _cuda_count
    if _cuda_count is not None and not force:
        return _cuda_count

    count = 0
    try:
        from .cuda_dlls import ensure_cuda_dlls
        ensure_cuda_dlls()
        import ctranslate2
        count = int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - absence of CUDA is a normal outcome here
        count = 0

    _cuda_count = count
    return count


def has_cuda() -> bool:
    return cuda_device_count() > 0


def resolve_device(requested: str) -> str:
    """Clamp a requested device to something this machine can actually run."""
    req = (requested or "auto").strip().lower()
    if req in ("auto", ""):
        return "cuda" if has_cuda() else "cpu"
    if req == "cuda" and not has_cuda():
        return "cpu"
    return "cpu" if req not in DEVICES else req


def resolve_compute_type(requested: str, device: str) -> str:
    """Pick a compute type valid for ``device``.

    float16 on CPU is slow and numerically poor for Whisper, so CPU is steered to
    int8, which is both the fastest and the most accurate CPU option in CTranslate2.
    """
    req = (requested or "auto").strip().lower()
    if device == "cpu":
        if req in ("auto", "float16", "int8_float16"):
            return "int8"
        return req if req in COMPUTE_TYPES else "int8"
    if req == "auto":
        return "float16"
    return req if req in COMPUTE_TYPES else "float16"
