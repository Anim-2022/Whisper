"""Model-reuse behaviour of WhisperEngine, including the CUDA -> CPU fallback.

Uses a fake faster_whisper module, so no CTranslate2, GPU or weights are needed.
"""
from __future__ import annotations

import sys
import types

import pytest

from whisper_engine.config import TranscriptionConfig
from whisper_engine.engine import WhisperEngine


class FakeModel:
    instances = 0

    def __init__(self, path, device=None, compute_type=None, local_files_only=None):
        FakeModel.instances += 1
        self.device = device
        self.compute_type = compute_type


class FailingOnCuda(FakeModel):
    """Mimics a machine where the CUDA libraries cannot be brought up."""

    def __init__(self, path, device=None, compute_type=None, local_files_only=None):
        if device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        super().__init__(path, device, compute_type, local_files_only)


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    """Install a stub faster_whisper module and return a setter for its model class."""
    module = types.ModuleType("faster_whisper")

    class Batched:
        def __init__(self, model=None):
            self.model = model

    module.BatchedInferencePipeline = Batched
    module.WhisperModel = FakeModel
    module.__version__ = "1.2.1-fake"
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    def use(model_cls):
        module.WhisperModel = model_cls
        FakeModel.instances = 0

    use(FakeModel)
    return use


@pytest.fixture
def model_dir(fake_ct2_dir, monkeypatch, tmp_path):
    d = fake_ct2_dir("whisper-medium", root=tmp_path / "ct2")
    monkeypatch.setattr("whisper_engine.engine.resolve_ct2_model_dir", lambda _id: d)
    return d


@pytest.fixture
def pretend_cuda(monkeypatch):
    """Let a config keep device="cuda" on a machine without a GPU.

    Without this the fallback tests silently test nothing: `resolve_device`
    rewrites "cuda" to "cpu" up front on a CI runner, the fake model constructor
    never sees "cuda", and no fallback is ever triggered.
    """
    monkeypatch.setattr("whisper_engine.device.has_cuda", lambda: True)


def cfg(**kw):
    kw.setdefault("model_id", "whisper-medium")
    return TranscriptionConfig(**kw)


def test_model_is_loaded_once_for_the_same_config(fake_faster_whisper, model_dir):
    engine = WhisperEngine()
    config = cfg(device="cpu")
    engine.load(config)
    engine.load(config)
    engine.load(config)
    assert FakeModel.instances == 1


@pytest.mark.parametrize("changed", [
    {"batched": False},
    {"compute_type": "float32"},
])
def test_changing_the_request_reloads(fake_faster_whisper, model_dir, changed):
    engine = WhisperEngine()
    engine.load(cfg(device="cpu"))
    engine.load(cfg(device="cpu", **changed))
    assert FakeModel.instances == 2


def test_cuda_failure_falls_back_to_cpu(fake_faster_whisper, model_dir, pretend_cuda):
    fake_faster_whisper(FailingOnCuda)
    logs = []
    engine = WhisperEngine(on_log=logs.append)

    engine.load(cfg(device="cuda", compute_type="float16"))

    assert engine.device == "cpu"
    assert engine.compute_type == "int8"
    assert any("CUDA initialization failed" in line for line in logs)


def test_the_fallback_is_not_retried_for_every_file(fake_faster_whisper, model_dir, pretend_cuda):
    """The regression this test exists for.

    Comparing the loaded state against config.device would leave the engine on
    cpu while the config still says cuda, so `_matches` would be false forever
    and every remaining file in a batch would repeat the slow failing CUDA load.
    """
    fake_faster_whisper(FailingOnCuda)
    engine = WhisperEngine()
    config = cfg(device="cuda", compute_type="float16")

    for _ in range(5):            # five files in one batch
        engine.load(config)

    # One failed CUDA attempt plus one successful CPU load, and nothing after.
    assert FakeModel.instances == 1
    assert engine.device == "cpu"


def test_unload_clears_the_request_too(fake_faster_whisper, model_dir):
    engine = WhisperEngine()
    config = cfg(device="cpu")
    engine.load(config)
    engine.unload()
    engine.load(config)
    assert FakeModel.instances == 2


def test_engine_info_reports_the_effective_device(fake_faster_whisper, model_dir, pretend_cuda):
    """Metadata must describe where it actually ran, not what was asked for."""
    fake_faster_whisper(FailingOnCuda)
    engine = WhisperEngine()
    engine.load(cfg(device="cuda", compute_type="float16"))
    info = engine.engine_info()
    assert info["device"] == "cpu"
    assert info["compute_type"] == "int8"


def test_non_cuda_failures_are_not_swallowed(fake_faster_whisper, model_dir):
    class AlwaysFails(FakeModel):
        def __init__(self, *a, **kw):
            raise RuntimeError("corrupt model file")

    fake_faster_whisper(AlwaysFails)
    with pytest.raises(RuntimeError, match="corrupt model file"):
        WhisperEngine().load(cfg(device="cpu"))
