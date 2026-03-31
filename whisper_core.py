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

from transformers import WhisperProcessor, WhisperForConditionalGeneration

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
    dtype: str = "float16" # "float16" or "float32"
    batch_size: int = 4
    max_new_tokens: int = 225
    decode_profile: Literal["balanced", "quality"] = "balanced"
    target_db: float = -20.0
    
    # Output
    save_srt: bool = False
    output_dir: Path = Path("./args_output")

    def __post_init__(self):
        # Auto-adjust device/dtype
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        
        if self.device == "cpu":
            self.dtype = "float32"

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
        return torch.float16 if self.dtype == "float16" else torch.float32

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
        
        # VAD
        self.vad_model = None
        self.vad_utils = None
        self.supports_attention_mask = True
        
        self.current_config = None

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
        raw_path = Path(model_path)
        if raw_path.exists():
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
                return max(snapshot_dirs, key=lambda path: path.stat().st_mtime)

        raise FileNotFoundError(
            f"Model '{model_path}' was not found locally. Choose a local model from the GUI "
            f"or add it under {self.models_root()}."
        )

    def resolve_local_vad_repo(self) -> Optional[Path]:
        for candidate in self.vad_repo_hints():
            if (candidate / "hubconf.py").exists():
                return candidate

        models_root = self.models_root()
        recursive_hits = sorted(
            path.parent
            for path in models_root.rglob("hubconf.py")
            if "silero-vad" in str(path.parent).lower()
        )
        if recursive_hits:
            return recursive_hits[0]

        hub_dir = Path(torch.hub.get_dir())
        cache_hits = sorted(path for path in hub_dir.glob("*silero-vad*") if (path / "hubconf.py").exists())
        return cache_hits[-1] if cache_hits else None

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
            self.processor = WhisperProcessor.from_pretrained(
                str(resolved_model_path),
                local_files_only=True,
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(resolved_model_path),
                torch_dtype=config.torch_dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).to(config.device)
            self.model.eval()
            self.log("Model loaded successfully.")
        except Exception as e:
            self.log(f"Error loading model: {e}")
            raise
    
    def load_vad_model(self):
        if self.vad_model is not None:
            return
            
        self.log("Loading Silero VAD model...")
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
            self.vad_utils = utils
            self.log("Silero VAD loaded.")
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

        audio = np.clip(audio, -1.0, 1.0)
        
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
            self.load_vad_model()
            
        if not self.vad_model or not self.vad_utils:
            self.log("VAD unavailable, skipping.")
            return []

        (get_speech_ts, _, _, _, _) = self.vad_utils
        
        # Silero expects tensor
        try:
            tensor_wav = torch.from_numpy(audio).float()
            
            # get_speech_timestamps returns list of dicts: [{'start': int, 'end': int}, ...]
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
            self.log("Running VAD (Silero)...")
            segments = self.get_speech_timestamps(audio, sr, config)
            total_duration = sum((e - s) for s, e in segments) / sr
            self.log(f"VAD found {len(segments)} segments. Total speech duration: {total_duration:.2f}s")
        
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
        
        # Check punctuation sentence boundary
        if a.rstrip().endswith(('.', '!', '?', '。', '！', '？')):
            return a.rstrip() + " " + b.lstrip()
            
        aw = a.split()
        bw = b.split()
        max_k = min(len(aw), len(bw), max_overlap_words)
        
        # Word overlap
        for k in range(max_k, 0, -1):
            aw_end = [w.lower().strip('.,!?;:') for w in aw[-k:]]
            bw_start = [w.lower().strip('.,!?;:') for w in bw[:k]]
            if aw_end == bw_start:
                return a + " " + " ".join(bw[k:])

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
                mid = len(chunk) // 2
                left = self.transcribe_chunk_safe(chunk[:mid], sr, config)
                right = self.transcribe_chunk_safe(chunk[mid:], sr, config)
                return self.stitch_text(left, right)
            raise

    def _run_model(self, chunks: List[np.ndarray], sr: int, config: TranscriptionConfig) -> List[str]:
        """Raw model generation."""
        if not chunks: return []

        forced_ids = None if config.auto_lang else self.processor.get_decoder_prompt_ids(
            language=config.lang, task="transcribe"
        )

        decode_kwargs = {
            "forced_decoder_ids": forced_ids,
            "max_new_tokens": config.max_new_tokens,
            "temperature": 0.0,
            "do_sample": False,
        }
        if config.decode_profile == "quality":
            decode_kwargs["num_beams"] = 5

        with torch.inference_mode():
            try:
                processed = self.processor(
                    chunks,
                    sampling_rate=sr,
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
            except TypeError as e:
                if "return_attention_mask" not in str(e).lower():
                    raise
                self.log("Whisper processor does not support return_attention_mask in this transformers build. Retrying.")
                processed = self.processor(
                    chunks,
                    sampling_rate=sr,
                    padding=True,
                    return_tensors="pt"
                )
            input_features = processed.input_features.to(config.device, dtype=config.torch_dtype)
            attention_mask = getattr(processed, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(config.device)

            generate_kwargs = dict(decode_kwargs)
            if attention_mask is not None and self.supports_attention_mask:
                generate_kwargs["attention_mask"] = attention_mask

            try:
                predicted_ids = self.model.generate(input_features, **generate_kwargs)
            except TypeError as e:
                if "attention_mask" not in str(e).lower() or not self.supports_attention_mask:
                    raise
                self.supports_attention_mask = False
                self.log("Whisper generate() does not support attention_mask in this transformers build. Retrying.")
                generate_kwargs.pop("attention_mask", None)
                predicted_ids = self.model.generate(input_features, **generate_kwargs)

            texts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return list(texts)

    def get_effective_batch_size(self, config: TranscriptionConfig) -> int:
        if config.decode_profile == "quality":
            return max(1, min(config.batch_size, 2))
        return max(1, config.batch_size)

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

                # Prepare Segments
                segments = self.prepare_segments(audio, sr, config)
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
                    
                    # File-level progress
                    file_percent = min(1.0, (i + len(batch)) / total_chunks)
                    overall_percent = (idx + file_percent) / overall_total
                    pct = int(file_percent * 100)
                    self.update_progress(overall_percent, f"Обработка: {pct}%")
                    
                    try:
                        texts = self._run_model(batch, sr, config)
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
                
                # Stitching/Aggregation
                self.log("Finalizing text...")
                final_segments = []
                for j, ((s, e), text) in enumerate(zip(segments, all_texts)):
                    text = text.strip()
                    if not text:
                        continue
                        
                    if not final_segments:
                        final_segments.append([text, s/sr, e/sr])
                    else:
                        prev_text, ps, pe = final_segments[-1]
                        # Only stitch if segments are very close or overlapping (sliding window)
                        # or if previous doesn't end with punctuation
                        time_gap = (s/sr) - pe
                        
                        if time_gap < 0.5 or not prev_text.rstrip().endswith(('.', '!', '?', '。', '！', '？')):
                            joined = self.stitch_text(prev_text, text)
                            final_segments[-1] = [joined, ps, e/sr]
                        else:
                            final_segments.append([text, s/sr, e/sr])
                
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
                            # format time helper
                            def fmt(sec):
                                ms = int((sec % 1) * 1000)
                                m, s_ = divmod(int(sec), 60)
                                h, m = divmod(m, 60)
                                return f"{h:02d}:{m:02d}:{s_:02d},{ms:03d}"
                            f.write(f"{k}\n{fmt(s)} --> {fmt(e)}\n{t}\n\n")
                    self.log(f"Saved SRT: {srt_path.name}")
                    
            except Exception as e:
                self.log(f"Error processing {file_path.name}: {e}")
                import traceback
                self.log(traceback.format_exc())
        
        self.update_progress(1.0, "Done.")
        if not self.stop_requested:
            self.log("All tasks completed.")
