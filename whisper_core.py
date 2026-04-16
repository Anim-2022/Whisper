# -*- coding: utf-8 -*-
"""
Core engine for Whisper GUI App.
Combines VAD (Silero), Batching, and OOM handling from start2.2.py with clean structure of run2.py.
"""
import os
import sys
import time
import logging
import gc
import re
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Callable, Literal

import numpy as np
import torch
import librosa
import soundfile as sf

# VAD: using Silero via torch.hub (no compilation needed)
HAS_VAD = True

from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline


def _fmt_srt_time(sec: float) -> str:
    """SRT timestamp format: HH:MM:SS,mmm (always 3-digit ms, comma separator)."""
    if sec is None or sec < 0:
        sec = 0.0
    ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:
        ms = 0
        sec_int = int(sec) + 1
    else:
        sec_int = int(sec)
    m, s_ = divmod(sec_int, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s_:02d},{ms:03d}"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class TranscriptionConfig:
    model_path: str = "openai/whisper-medium"
    lang: str = "de"
    auto_lang: bool = False
    
    # Segmentation
    chunk_sec: float = 20.0
    overlap_sec: float = 3.0
    min_segment_sec: float = 1.5
    max_segment_sec: float = 28.0
    
    # VAD
    use_vad: bool = True
    vad_threshold: float = 0.5  # Replaces aggressiveness
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 100
    vad_merge_gap_sec: float = 0.25
    pad_sec: float = 0.2
    
    # Model / Compute
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # "auto" picks bf16 on Ampere+ (cc>=8.0), fp16 on older CUDA, fp32 on CPU.
    # Explicit values: "float16" | "bfloat16" | "float32".
    dtype: str = "auto"
    batch_size: int = 4
    max_new_tokens: int = 225
    decode_profile: Literal["balanced", "quality"] = "balanced"
    target_db: float = -20.0
    # Compile model.forward with torch.compile. Adds ~30-60s warmup on the first
    # batch but yields +10-25% throughput on Ampere/Ada with SDPA. Off by default
    # because recompiles on shape changes can spam logs.
    use_compile: bool = False
    # Use HF transformers' built-in long-form ASR pipeline instead of the manual
    # VAD->chunk->batch->stitch loop. The pipeline packs short segments into 30 s
    # mel windows (so the encoder isn't wasted on padding), uses Whisper-aware
    # token-level stitching with timestamps, and handles language/task internally.
    # Off by default for rollback safety; enable once validated on your corpus.
    use_hf_pipeline: bool = False
    # Stride (seconds) on each side of the 30 s pipeline window. 5 s is the HF
    # default and matches Whisper's training stride.
    pipeline_stride_sec: float = 5.0
    
    # Output
    save_srt: bool = False
    output_dir: Path = Path("./args_output")

    def __post_init__(self):
        # Auto-adjust device/dtype
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        if self.device == "cpu":
            # bf16/fp16 inference is unstable/slow on CPU for Whisper; force fp32.
            self.dtype = "float32"
        elif self.dtype == "auto":
            # Prefer bf16 on Ampere+ (compute capability 8.0+): same speed as fp16,
            # much wider exponent range -> no NaNs in encoder LayerNorms on noisy
            # audio. Fall back to fp16 on Turing/Volta and older.
            try:
                major, _ = torch.cuda.get_device_capability()
            except Exception:
                major = 0
            self.dtype = "bfloat16" if major >= 8 else "float16"

        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"dtype must be one of float16/bfloat16/float32/auto, got {self.dtype!r}")

        if self.chunk_sec <= 0:
            raise ValueError("chunk_sec must be positive")
        if self.overlap_sec < 0 or self.overlap_sec >= self.chunk_sec:
            raise ValueError("overlap_sec must be non-negative and smaller than chunk_sec")
        if self.min_segment_sec > self.max_segment_sec:
            raise ValueError("min_segment_sec must be <= max_segment_sec")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.vad_merge_gap_sec < 0:
            raise ValueError("vad_merge_gap_sec must be >= 0")
        if self.target_db > 0:
            raise ValueError("target_db should be negative (dB)")
        if self.decode_profile not in {"balanced", "quality"}:
            raise ValueError("decode_profile must be 'balanced' or 'quality'")

    @property
    def torch_dtype(self):
        return {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }[self.dtype]

# -----------------------------------------------------------------------------
# Core Engine Class
# -----------------------------------------------------------------------------
class WhisperCore:
    def __init__(self, 
                 on_log: Optional[Callable[[str], None]] = None,
                 on_progress: Optional[Callable[[float, str], None]] = None):
        """
        Args:
            on_log: Callback accepting a log string.
            on_progress: Callback accepting (float percent 0-1, str status_text).
        """
        self.on_log = on_log
        self.on_progress = on_progress
        self.stop_requested = False
        self.processor = None
        self.model = None
        # HF asr pipeline (lazily built on first use when config.use_hf_pipeline=True)
        self.asr_pipeline = None

        # VAD
        self.vad_model = None
        self.vad_utils = None

        self.current_config = None
        
        # Internal cache for faster subsequent loads or checks
        self._cache_model_path = {}
        self._cache_vad_path = None

    def models_root(self) -> Path:
        return Path(__file__).parent / "models"

    def vad_repo_hints(self) -> List[Path]:
        root = self.models_root()
        return [
            root / "silero-vad",
            root / "snakers4-silero-vad",
            root / "repos" / "silero-vad",
            root / "repos" / "snakers4-silero-vad",
        ]

    def resolve_local_model_path(self, model_path: str) -> Path:
        if model_path in self._cache_model_path:
            return self._cache_model_path[model_path]

        raw_path = Path(model_path)
        if raw_path.exists():
            self._cache_model_path[model_path] = raw_path
            return raw_path

        cache_dir = self.models_root() / f"models--{model_path.replace('/', '--')}"
        snapshots_dir = cache_dir / "snapshots"
        ref_main = cache_dir / "refs" / "main"

        if ref_main.exists():
            snapshot_name = ref_main.read_text(encoding="utf-8").strip()
            if snapshot_name:
                snapshot_dir = snapshots_dir / snapshot_name
                if snapshot_dir.exists():
                    return snapshot_dir

        if snapshots_dir.exists():
            snapshot_dirs = [path for path in snapshots_dir.iterdir() if path.is_dir()]
            if snapshot_dirs:
                res = max(snapshot_dirs, key=lambda path: path.stat().st_mtime)
                self._cache_model_path[model_path] = res
                return res

        raise FileNotFoundError(
            f"Model '{model_path}' was not found locally. Choose a local model from the GUI "
            f"or add it under {self.models_root()}."
        )

    def resolve_local_vad_repo(self) -> Optional[Path]:
        if self._cache_vad_path and self._cache_vad_path.exists():
            return self._cache_vad_path

        for candidate in self.vad_repo_hints():
            if (candidate / "hubconf.py").exists():
                self._cache_vad_path = candidate
                return candidate

        models_root = self.models_root()
        # Optimization: Don't use rglob on models_root as it scans everything.
        # Just look into suspected sub-directories.
        search_dirs = [models_root, models_root / "repos", models_root / "checkpoints"]
        for sd in search_dirs:
            if not sd.exists(): continue
            for entry in sd.iterdir():
                if entry.is_dir() and "silero-vad" in entry.name.lower():
                    if (entry / "hubconf.py").exists():
                        self._cache_vad_path = entry
                        return entry

        # Further fallback: if we really have to search, look for hubconf.py in a shallower way.
        # Actually torch.hub.load usually expects it at the top or one level deep.
        hub_dir = Path(torch.hub.get_dir())
        if hub_dir.exists():
            cache_hits = sorted(path for path in hub_dir.glob("*silero-vad*") if (path / "hubconf.py").exists())
            if cache_hits:
                self._cache_vad_path = cache_hits[-1]
                return self._cache_vad_path
        
        return None

    def log(self, msg: str):
        if self.on_log:
            self.on_log(msg)
        else:
            print(msg)

    def update_progress(self, percent: float, status: str):
        if self.on_progress:
            self.on_progress(percent, status)

    def request_stop(self):
        self.stop_requested = True
        self.log("!!! STOP REQUESTED !!!")

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------
    def load_model(self, config: TranscriptionConfig):
        self.current_config = config
        self.log(f"Loading model: {config.model_path}")
        self.log(f"Device: {config.device} ({config.dtype})")
        
        try:
            resolved_model_path = self.resolve_local_model_path(config.model_path)
            self.log(f"Resolved local model path: {resolved_model_path}")
            
            self.log("Step 1/2: Loading processor...")
            self.processor = WhisperProcessor.from_pretrained(
                str(resolved_model_path),
                local_files_only=True,
            )
            
            self.log("Step 2/2: Loading weights (this may take a moment)...")

            # Pick best attention implementation: FA2 if installed, otherwise SDPA
            # (built into torch>=2.0). Falls back to "eager" if the runtime rejects
            # the chosen impl on this model/dtype combo.
            attn_impl = "sdpa"
            if config.device != "cpu" and config.dtype in ("float16", "bfloat16"):
                try:
                    import flash_attn  # noqa: F401
                    attn_impl = "flash_attention_2"
                except ImportError:
                    pass

            # device_map="auto" is only useful for multi-GPU or CPU offload of large
            # models. On a single GPU it adds accelerate hooks that block plain .to()
            # movement and can spuriously offload layers. Restrict accordingly.
            multi_gpu = (
                config.device != "cpu"
                and torch.cuda.is_available()
                and torch.cuda.device_count() > 1
            )
            has_accelerate = False
            if multi_gpu:
                try:
                    import accelerate  # noqa: F401
                    has_accelerate = True
                except ImportError:
                    pass

            load_kwargs = {
                "torch_dtype": config.torch_dtype,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
                "attn_implementation": attn_impl,
            }
            if has_accelerate and multi_gpu:
                load_kwargs["device_map"] = "auto"

            try:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    str(resolved_model_path),
                    **load_kwargs,
                )
                self.log(f"Attention implementation: {attn_impl}")
            except (ValueError, ImportError, RuntimeError) as e:
                # FA2/SDPA may be unavailable for some model+dtype combos; fall back.
                if attn_impl != "eager":
                    self.log(f"attn_implementation={attn_impl!r} failed ({e}); retrying with 'eager'.")
                    load_kwargs["attn_implementation"] = "eager"
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        str(resolved_model_path),
                        **load_kwargs,
                    )
                else:
                    raise

            # If not using device_map, manually move to device.
            if "device_map" not in load_kwargs:
                self.log(f"Moving model to {config.device}...")
                self.model = self.model.to(config.device)

            self.model.eval()

            # Optional torch.compile pass. Wrap model.forward, not the whole module —
            # generate() has Python-side control flow that defeats fullgraph=True.
            # mode='reduce-overhead' targets the per-step decoder forward, which is
            # what we re-enter on every generated token.
            #
            # The default Inductor backend requires Triton, which is not packaged
            # with PyTorch on Windows. Silently skip compile when Triton is missing
            # rather than crashing on the first forward pass.
            if config.use_compile and config.device != "cpu" and hasattr(torch, "compile"):
                try:
                    import triton  # noqa: F401
                    triton_ok = True
                except ImportError:
                    triton_ok = False

                if not triton_ok:
                    self.log("torch.compile requested but Triton is not installed "
                             "(install 'triton' wheel for Windows). Skipping compile.")
                else:
                    try:
                        self.model.forward = torch.compile(
                            self.model.forward,
                            mode="reduce-overhead",
                            fullgraph=False,
                            dynamic=True,   # batch shape varies; avoids constant recompile
                        )
                        self.log("torch.compile enabled (mode=reduce-overhead, dynamic).")
                    except Exception as e:
                        self.log(f"torch.compile setup failed ({e}); continuing without compile.")
            self.log("Model loading complete.")
        except Exception as e:
            self.log(f"Error loading model: {e}")
            raise
    
    def load_vad_model(self, device: str = "cpu"):
        if self.vad_model is not None:
            # If already loaded on different device, move it
            if str(self.vad_model.device) != device and device != "cpu":
                self.vad_model.to(device)
            return
            
        self.log(f"Loading Silero VAD model to {device}...")
        try:
            local_vad_repo = self.resolve_local_vad_repo()
            if local_vad_repo is None:
                preferred_path = self.vad_repo_hints()[0]
                self.log(
                    "Silero VAD was not found locally. "
                    f"Place the repository at '{preferred_path}' to use VAD offline."
                )
                self.vad_model = None
                self.vad_utils = None
                return

            self.log(f"Resolved local VAD repo: {local_vad_repo}")
            self.vad_model, utils = torch.hub.load(
                repo_or_dir=str(local_vad_repo),
                model='silero_vad',
                source='local',
                force_reload=False,
                onnx=False,
            )
            if device != "cpu":
                self.vad_model.to(device)
            self.vad_utils = utils
            self.log(f"Silero VAD loaded on {device}.")
        except Exception as e:
            self.log(f"Error loading VAD: {e}")
            self.vad_model = None
            self.vad_utils = None

    # -------------------------------------------------------------------------
    # Audio Utils
    # -------------------------------------------------------------------------
    def normalize_audio(
        self,
        audio: np.ndarray,
        sr: int,
        target_sr: int = 16000,
        target_db: float = -20.0,
    ) -> Tuple[np.ndarray, int]:
        # Mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # DC offset
        audio = audio - np.mean(audio)

        # RMS normalization. Skip almost-silent inputs to avoid amplifying noise.
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-6:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)

        # Peak rescale (not clip): preserves dynamics of percussive/dynamic recordings.
        # Hard clipping at 1.0 after RMS gain produces flat-tops -> intermodulation
        # distortion that the encoder hears as garbled phonemes.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.99:
            audio = audio * (0.99 / peak)
        
        # Resample
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="polyphase")
            sr = target_sr
            
        return audio, sr

    def merge_segments(self, segments: List[Tuple[int, int]], max_gap_samples: int) -> List[Tuple[int, int]]:
        if not segments:
            return []

        merged = [list(segments[0])]
        for s, e in segments[1:]:
            _, prev_end = merged[-1]
            if s <= prev_end + max_gap_samples:
                merged[-1][1] = max(prev_end, e)
            else:
                merged.append([s, e])
        return [tuple(x) for x in merged]

    # -------------------------------------------------------------------------
    # VAD & Segmentation
    # -------------------------------------------------------------------------
    def get_speech_timestamps(self, audio: np.ndarray, sr: int, config: TranscriptionConfig) -> List[Tuple[int, int]]:
        if not self.vad_model:
            self.load_vad_model(device=config.device)
            
        if not self.vad_model or not self.vad_utils:
            self.log("VAD unavailable, skipping.")
            return []

        (get_speech_ts, _, _, _, _) = self.vad_utils
        
        # Silero expects tensor
        try:
            device = next(self.vad_model.parameters()).device
            tensor_wav = torch.from_numpy(audio).float().to(device)
            
            # get_speech_timestamps returns list of dicts: [{'start': int, 'end': int}, ...]
            with torch.inference_mode():
                timestamps = get_speech_ts(
                    tensor_wav, 
                    self.vad_model, 
                    threshold=config.vad_threshold,
                    sampling_rate=sr,
                    min_speech_duration_ms=config.vad_min_speech_ms,
                    min_silence_duration_ms=config.vad_min_silence_ms
                )
            
            # Convert to list of tuples
            segments = [(ts['start'], ts['end']) for ts in timestamps]
            
            # Apply padding and merge close segments so short pauses don't cut words.
            pad_samples = int(config.pad_sec * sr)
            final = []
            for s, e in segments:
                final.append((max(0, s - pad_samples), min(len(audio), e + pad_samples)))

            merge_gap_samples = int(config.vad_merge_gap_sec * sr)
            return self.merge_segments(final, merge_gap_samples)
            
            
        except Exception as e:
            self.log(f"VAD Execution error: {e}")
            return []

    def prepare_segments(self, audio: np.ndarray, sr: int, config: TranscriptionConfig) -> List[Tuple[int, int]]:
        # 1. Try VAD
        segments = []
        if config.use_vad:
            if self.stop_requested:
                return []
            self.log(f"Running VAD (Silero) on {config.device}...")
            start_time = time.time()
            segments = self.get_speech_timestamps(audio, sr, config)
            if self.stop_requested:
                return []
            duration = time.time() - start_time
            total_duration = sum((e - s) for s, e in segments) / sr
            self.log(f"VAD found {len(segments)} segments took {duration:.2f}s. Total speech duration: {total_duration:.2f}s")
        
        # 2. Fallback to sliding window
        if not segments:
            if config.use_vad:
                self.log("VAD found nothing or was too restrictive. Falling back to sliding window.")
            chunk_samples = int(config.chunk_sec * sr)
            overlap_samples = int(config.overlap_sec * sr)
            step = max(1, chunk_samples - overlap_samples)
            
            for start in range(0, len(audio), step):
                end = min(start + chunk_samples, len(audio))
                segments.append((start, end))
                if end >= len(audio):
                    break
            self.log(f"Created {len(segments)} chunks using sliding window.")
        
        # 3. Aggregate small segments
        min_samples = int(config.min_segment_sec * sr)
        aggregated = []
        for s, e in segments:
            if (e - s) < min_samples and aggregated:
                ps, pe = aggregated[-1]
                aggregated[-1] = (ps, e)
            else:
                aggregated.append((s, e))
        segments = aggregated

        # 4. Split long segments with overlap
        final = []
        max_samples = int(config.max_segment_sec * sr)
        overlap_samples = int(config.overlap_sec * sr)
        
        for s, e in segments:
            if (e - s) > max_samples:
                cur = s
                while cur < e:
                    end = min(cur + max_samples, e)
                    final.append((cur, end))
                    if end >= e:
                        break
                    # Move forward but keep some overlap for stitching
                    cur = end - overlap_samples
            else:
                final.append((s, e))
                
        return final

    # -------------------------------------------------------------------------
    # Transcription & Stitching
    # -------------------------------------------------------------------------
    def stitch_text(self, a: str, b: str, max_overlap_words: int = 40) -> str:
        """Clean stitching with exact and fuzzy overlap detection."""
        if not a: return b
        if not b: return a
        
        # Word overlap detection
        aw = a.split()
        bw = b.split()
        max_k = min(len(aw), len(bw), max_overlap_words)
        
        for k in range(max_k, 0, -1):
            aw_end = [w.lower().strip('.,!?;:') for w in aw[-k:]]
            bw_start = [w.lower().strip('.,!?;:') for w in bw[:k]]
            if aw_end == bw_start:
                return a + " " + " ".join(bw[k:])

        # If a ends with punctuation, we still check for substantial word overlap 
        # because Whisper might repeat a whole sentence in the next chunk.
        if a.rstrip().endswith(('.', '!', '?', '。', '！', '？')):
            # If last 3 words of A are same as first 3 words of B, it's likely a repeat
            if len(aw) >= 3 and len(bw) >= 3:
                if [w.lower().strip('.,!?;:') for w in aw[-3:]] == [w.lower().strip('.,!?;:') for w in bw[:3]]:
                    # Use fuzzy matcher to find exact cut point
                    pass # Continue to fuzzy logic below
            else:
                return a.rstrip() + " " + b.lstrip()

        left_edge = a[-500:] if len(a) > 500 else a
        right_edge = b[:500] if len(b) > 500 else b
        match = SequenceMatcher(None, left_edge, right_edge).find_longest_match(
            0, len(left_edge), 0, len(right_edge)
        )
        if match.size > 20:
            left_keep = len(a) - len(left_edge) + match.a + match.size
            return a[:left_keep] + right_edge[match.b + match.size:]
        
        # Fallback
        return a.rstrip() + " " + b.lstrip()

    def transcribe_chunk_safe(self, chunk: np.ndarray, sr: int, config: TranscriptionConfig) -> str:
        """Transcribe single chunk with OOM handling logic."""
        # Simple recursion for OOM
        try:
            return self._run_model([chunk], sr, config)[0]
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and len(chunk) > sr:
                self.log("OOM detected in single chunk, splitting...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Split with 0.5s overlap on each side so stitch_text has real shared
                # acoustic content to align — splitting at exact midpoint cuts a word.
                overlap = sr // 2
                mid = len(chunk) // 2
                left_end   = min(len(chunk), mid + overlap)
                right_start = max(0, mid - overlap)
                left  = self.transcribe_chunk_safe(chunk[:left_end], sr, config)
                right = self.transcribe_chunk_safe(chunk[right_start:], sr, config)
                return self.stitch_text(left, right)
            raise

    def _run_model(self, chunks: List[np.ndarray], sr: int, config: TranscriptionConfig) -> List[str]:
        """Raw model generation."""
        if not chunks: return []

        # Native language/task kwargs (replaces deprecated forced_decoder_ids path).
        decode_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 5,
        }
        if not config.auto_lang:
            decode_kwargs["language"] = config.lang
            decode_kwargs["task"] = "transcribe"
        if config.decode_profile == "quality":
            decode_kwargs["num_beams"] = 5
            decode_kwargs["length_penalty"] = 1.0
            decode_kwargs["early_stopping"] = False

        with torch.inference_mode():
            processed = self.processor(
                chunks,
                sampling_rate=sr,
                return_tensors="pt",
            )
            input_features = processed.input_features.to(config.device, dtype=config.torch_dtype)

            # Note: do NOT forward processor's attention_mask to Whisper.generate() —
            # it is a waveform-level mask, not aligned to the encoder's mel frames.
            # Whisper's encoder operates on a fixed 30 s mel canvas and handles padding internally.
            predicted_ids = self.model.generate(input_features, **decode_kwargs)

            texts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return list(texts)

    def get_effective_batch_size(self, config: TranscriptionConfig) -> int:
        if config.decode_profile == "quality":
            return max(1, min(config.batch_size, 2))
        return max(1, config.batch_size)

    def _ensure_pipeline(self, config: TranscriptionConfig):
        """Build the HF ASR pipeline lazily after the model is loaded."""
        if self.asr_pipeline is not None:
            return
        if self.model is None or self.processor is None:
            raise RuntimeError("Model must be loaded before building HF pipeline")
        # device for pipeline: torch.device for cuda, -1 for CPU
        if config.device == "cpu":
            pipe_device = -1
        else:
            pipe_device = torch.device(config.device)
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=config.torch_dtype,
            device=pipe_device,
        )
        self.log("HF ASR pipeline initialized.")

    def _run_pipeline(self, audio: np.ndarray, sr: int, config: TranscriptionConfig
                     ) -> Tuple[str, List[dict]]:
        """One-shot long-form transcription via HF pipeline.
        Returns (full_text, chunks) where chunks is a list of
        {'text': str, 'timestamp': (start_sec, end_sec)} dicts.
        """
        self._ensure_pipeline(config)

        gen_kwargs = {
            "no_repeat_ngram_size": 5,
            "repetition_penalty": 1.1,
            "max_new_tokens": config.max_new_tokens,
        }
        if not config.auto_lang:
            gen_kwargs["language"] = config.lang
            gen_kwargs["task"] = "transcribe"
        if config.decode_profile == "quality":
            gen_kwargs["num_beams"] = 5
            gen_kwargs["length_penalty"] = 1.0
            gen_kwargs["early_stopping"] = False

        bs = self.get_effective_batch_size(config)

        with torch.inference_mode():
            result = self.asr_pipeline(
                {"array": audio.astype(np.float32, copy=False), "sampling_rate": sr},
                chunk_length_s=30.0,
                stride_length_s=(config.pipeline_stride_sec, config.pipeline_stride_sec),
                batch_size=bs,
                generate_kwargs=gen_kwargs,
                return_timestamps=True,
            )
        text = (result.get("text") or "").strip()
        chunks = result.get("chunks") or []
        return text, chunks

    def process_files(self, file_paths: List[Path], config: TranscriptionConfig):
        """Main entry point to process a list of files."""
        self.stop_requested = False
        self.current_config = config
        
        if not self.model:
            self.load_model(config)
            
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        overall_total = len(file_paths)
        
        for idx, file_path in enumerate(file_paths):
            if self.stop_requested:
                break
                
            self.update_progress(idx / overall_total, f"Processing {file_path.name}...")
            self.log(f"--- Started: {file_path.name} ---")
            
            try:
                # Load Audio
                try:
                    audio, sr = sf.read(str(file_path))
                    audio, sr = self.normalize_audio(audio, sr, target_db=config.target_db)
                except Exception as e:
                    self.log(f"Failed to read audio {file_path.name}: {e}")
                    continue

                # ----- HF pipeline path (single-shot long-form) -----
                if config.use_hf_pipeline:
                    self.update_progress((idx + 0.05) / overall_total,
                                         f"Pipeline: {file_path.name}")
                    try:
                        text, pipe_chunks = self._run_pipeline(audio, sr, config)
                    except RuntimeError as e:
                        if "out of memory" not in str(e).lower():
                            raise
                        # OOM in pipeline -> drop pipeline state, free VRAM, fall back
                        # to the manual VAD/batch path for this file.
                        self.log("OOM in HF pipeline. Falling back to manual loop for this file...")
                        self.asr_pipeline = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        # Save TXT
                        stem = file_path.stem
                        txt_path = config.output_dir / f"{stem}.txt"
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        self.log(f"Saved TXT: {txt_path.name}")
                        # Save SRT from pipeline chunks
                        if config.save_srt and pipe_chunks:
                            srt_path = config.output_dir / f"{stem}.srt"
                            with open(srt_path, "w", encoding="utf-8") as f:
                                for k, ch in enumerate(pipe_chunks, 1):
                                    ts = ch.get("timestamp") or (None, None)
                                    s_sec = ts[0] if ts[0] is not None else 0.0
                                    e_sec = ts[1] if ts[1] is not None else (s_sec + 1.0)
                                    f.write(f"{k}\n{_fmt_srt_time(s_sec)} --> {_fmt_srt_time(e_sec)}\n{(ch.get('text') or '').strip()}\n\n")
                            self.log(f"Saved SRT: {srt_path.name}")
                        self.update_progress((idx + 1.0) / overall_total,
                                             f"Done: {file_path.name}")
                        continue  # next file

                # ----- Manual VAD/batch path (fallback and default) -----
                # Prepare Segments
                segments = self.prepare_segments(audio, sr, config)
                if self.stop_requested:
                    self.log("Stopped by user.")
                    break
                if not segments:
                    self.log(f"No speech detected in {file_path.name}; skipping.")
                    continue
                chunks = [audio[s:e] for s, e in segments]
                
                # Transcribe
                all_texts = []
                total_chunks = len(chunks)
                
                # Batch processing
                bs = self.get_effective_batch_size(config)
                if config.decode_profile == "quality" and bs != config.batch_size:
                    self.log(f"Quality profile limiting batch size to {bs} for decoding stability.")
                for i in range(0, total_chunks, bs):
                    if self.stop_requested:
                        break
                    
                    batch = chunks[i : i + bs]
                    
                    # Pre-filter extremely quiet chunks to avoid hallucinations
                    valid_batch = []
                    valid_indices = []
                    for idx_b, chunk in enumerate(batch):
                        if np.max(np.abs(chunk)) > 0.005: 
                            valid_batch.append(chunk)
                            valid_indices.append(idx_b)
                    
                    # File-level progress
                    file_percent = min(1.0, (i + len(batch)) / total_chunks)
                    overall_percent = (idx + file_percent) / overall_total
                    pct = int(file_percent * 100)
                    self.update_progress(overall_percent, f"Обработка: {pct}%")
                    
                    try:
                        if valid_batch:
                            texts = self._run_model(valid_batch, sr, config)
                            # Map results back to full batch size with empty strings for skipped
                            full_texts = [""] * len(batch)
                            for v_idx, text in zip(valid_indices, texts):
                                full_texts[v_idx] = text
                            texts = full_texts
                        else:
                            texts = [""] * len(batch)
                            
                        all_texts.extend(texts)
                        # Log preview for debug
                        if texts and texts[0].strip():
                            full_text = texts[0].strip()
                            snippet = full_text[:100] + "..." if len(full_text) > 100 else full_text
                            self.log(f"  → {snippet}")
                        elif texts:
                            self.log(f"  → [тишина]")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            self.log("OOM in batch. Falling back to serial...")
                            gc.collect()
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            for c in batch:
                                all_texts.append(self.transcribe_chunk_safe(c, sr, config))
                        else:
                            raise

                if self.stop_requested:
                    self.log("Stopped by user.")
                    break
                
                # Stitching/Aggregation.
                # Two distinct cases that must NOT be conflated:
                #   (A) Acoustic overlap (s < previous_end_in_samples): segments share
                #       audio (sliding window or split-with-overlap path). Whisper will
                #       transcribe the shared region twice, so we need stitch_text() to
                #       detect and remove the duplicate words.
                #   (B) No overlap (consecutive VAD segments): just concatenate. Calling
                #       stitch_text on these is the bug — it can find a coincidental
                #       3-word match between unrelated sentences and DELETE the prefix
                #       of the second sentence.
                # The time_gap heuristic decides whether to merge into one final entry
                # (close in time / no terminal punctuation) or start a new entry.
                self.log("Finalizing text...")
                final_segments = []
                last_end_sample = -1  # end of previous accepted source segment, in samples
                for j, ((s, e), text) in enumerate(zip(segments, all_texts)):
                    text = text.strip()
                    if not text:
                        continue

                    overlapping = (s < last_end_sample)

                    if not final_segments:
                        final_segments.append([text, s/sr, e/sr])
                    else:
                        prev_text, ps, pe = final_segments[-1]
                        time_gap = (s/sr) - pe
                        close_in_time = time_gap < 0.5
                        no_terminal_punct = not prev_text.rstrip().endswith(('.', '!', '?', '。', '！', '？'))

                        if overlapping:
                            # Real shared audio -> dedupe via overlap matcher.
                            joined = self.stitch_text(prev_text, text)
                            final_segments[-1] = [joined, ps, e/sr]
                        elif close_in_time or no_terminal_punct:
                            # Adjacent but non-overlapping -> safe concatenate (no word loss).
                            final_segments[-1] = [
                                prev_text.rstrip() + " " + text.lstrip(),
                                ps,
                                e/sr,
                            ]
                        else:
                            final_segments.append([text, s/sr, e/sr])

                    last_end_sample = e
                
                # Convert back to tuples
                final_segments = [tuple(x) for x in final_segments]
                
                # Save
                stem = file_path.stem
                txt_path = config.output_dir / f"{stem}.txt"
                full_text = " ".join(t for t, _, _ in final_segments)
                
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                self.log(f"Saved TXT: {txt_path.name}")
                
                if config.save_srt:
                    srt_path = config.output_dir / f"{stem}.srt"
                    with open(srt_path, "w", encoding="utf-8") as f:
                        for k, (t, s, e) in enumerate(final_segments, 1):
                            f.write(f"{k}\n{_fmt_srt_time(s)} --> {_fmt_srt_time(e)}\n{t}\n\n")
                    self.log(f"Saved SRT: {srt_path.name}")
                    
            except Exception as e:
                self.log(f"Error processing {file_path.name}: {e}")
                import traceback
                self.log(traceback.format_exc())
        
        if self.stop_requested:
            self.update_progress(0.0, "Stopped")
        else:
            self.update_progress(1.0, "Done.")
            self.log("All tasks completed.")
