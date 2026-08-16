"""Make the CUDA runtime DLLs discoverable before CTranslate2 is imported.

Since Python 3.8, ``PATH`` is not consulted when resolving the dependencies of a
native extension module on Windows — only directories registered through
``os.add_dll_directory()`` (plus a few system locations) are searched. The
CTranslate2 wheel links against cuBLAS and cuDNN but ships neither, so without
this step ``import ctranslate2`` fails with a bare "DLL load failed".

Import order matters: :func:`ensure_cuda_dlls` must run *before* the first
``import ctranslate2`` (which ``faster_whisper`` performs at import time).

Two mechanisms are needed, and using only one is the classic failure mode:

* ``os.add_dll_directory()`` resolves the *static* dependencies of the extension
  module itself.
* Prepending to ``PATH`` covers libraries CTranslate2 opens *lazily* at first
  use — cuBLAS and cuDNN are loaded on the first encoder call via a bare-name
  ``LoadLibraryA``, which follows the standard search order and ignores the
  directories registered above. Skipping this yields a model that loads on CUDA,
  runs VAD, detects the language, and only then dies with
  "Library cublas64_12.dll is not found or cannot be loaded".

Search order is deliberate. The nvidia pip wheels come first because the
CTranslate2 4.8.x Windows build targets CUDA 12.8 / cuDNN 9.10; the DLLs bundled
with torch are older (CUDA 12.4 / cuDNN 9.1) and are only a last resort.

Mixing cuDNN versions is worse than having none: cuDNN 9 loads its sublibraries
(``cudnn_graph64_9.dll``, ``cudnn_engines_*``) lazily by bare name, so a 9.24
``cudnn64_9.dll`` paired with a 9.1 ``cudnn_graph64_9.dll`` crashes at the first
convolution. Therefore only the *first* directory providing a given library
family is registered.
"""
from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

# Library families we resolve, in priority order within each source.
_NVIDIA_WHEEL_SUBDIRS = (
    ("cudnn", "bin"),
    ("cublas", "bin"),
    ("cuda_runtime", "bin"),
    ("cuda_nvrtc", "bin"),
)

_registered: list[str] | None = None


def _site_packages() -> list[Path]:
    paths = []
    for key in ("purelib", "platlib"):
        try:
            p = sysconfig.get_paths().get(key)
        except Exception:  # noqa: BLE001 - sysconfig is best-effort here
            p = None
        if p:
            paths.append(Path(p))
    # Deduplicate while preserving order (purelib == platlib in a normal venv).
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _candidate_dirs() -> list[Path]:
    """Directories that may hold CUDA DLLs, most-preferred first."""
    dirs: list[Path] = []

    # 1. nvidia-* pip wheels: site-packages/nvidia/<lib>/bin
    for sp in _site_packages():
        nvidia_root = sp / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib, sub in _NVIDIA_WHEEL_SUBDIRS:
            d = nvidia_root / lib / sub
            if d.is_dir():
                dirs.append(d)

    # 2. A system CUDA toolkit, if the user has one.
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        d = Path(cuda_path) / "bin"
        if d.is_dir():
            dirs.append(d)

    # 3. torch's bundled DLLs. Older than CTranslate2's build target, so this is
    #    a fallback for environments that still have torch installed.
    for sp in _site_packages():
        d = sp / "torch" / "lib"
        if d.is_dir():
            dirs.append(d)

    return dirs


def ensure_cuda_dlls(force: bool = False) -> list[str]:
    """Register CUDA DLL directories with the loader. Returns what was registered.

    No-op on non-Windows platforms and on repeated calls. Never raises: a failure
    to register simply means CUDA will be unavailable and the caller falls back
    to CPU, which is a better outcome than an import-time crash.
    """
    global _registered
    if _registered is not None and not force:
        return _registered

    if sys.platform != "win32":
        _registered = []
        return _registered

    added: list[str] = []
    claimed: set[str] = set()
    for d in _candidate_dirs():
        # Only the first provider of a library family wins (see module docstring).
        family = d.parent.name if d.name == "bin" else d.name
        if family in claimed:
            continue
        try:
            os.add_dll_directory(str(d))
        except (OSError, AttributeError):
            continue
        claimed.add(family)
        added.append(str(d))

    if added:
        # Required for CTranslate2's lazy bare-name loads of cuBLAS/cuDNN; see
        # the module docstring. Prepend so our versions win over any system copy.
        existing = os.environ.get("PATH", "")
        prefix = os.pathsep.join(added)
        if prefix not in existing:
            os.environ["PATH"] = prefix + os.pathsep + existing

    _registered = added
    return _registered


def describe() -> str:
    """Human-readable summary for the log, so support takes ten seconds."""
    dirs = ensure_cuda_dlls()
    if sys.platform != "win32":
        return "CUDA DLL registration: not needed on this platform."
    if not dirs:
        return ("CUDA DLL registration: nothing found. Install the GPU extras "
                "(pip install -r requirements-gpu.txt) if you expect CUDA.")
    return "CUDA DLL directories registered:\n  " + "\n  ".join(dirs)
