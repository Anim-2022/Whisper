<div align="center">

# 🎙️ Whisper Unified

**Offline-first desktop GUI for local speech-to-text transcription**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Engine](https://img.shields.io/badge/Engine-faster--whisper-orange)](https://github.com/SYSTRAN/faster-whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078d7?logo=windows)](https://www.microsoft.com/)

[🇺🇦 Українська](README.uk.md) · [🇷🇺 Русский](README.ru.md) · [🇬🇧 English](#)

</div>

---

## ✨ Features

- **100% offline** — no API keys. The only step that uses the network is fetching the model once
- **faster-whisper (CTranslate2)** — 16–26× realtime with the default preset on a 16 GB consumer GPU, and 53–60× in the long-recording preset, which turns a 2 h 52 m file into about 2.5 minutes
- **Reads what your meetings actually produce** — `m4a`, `mp4`, `webm`, `mkv`, `opus` and more, via bundled FFmpeg. No system install needed
- **Silero VAD** — bundled with the engine; skips silence, which is both faster and less prone to hallucinated text
- **GPU / CPU** — CUDA acceleration with an automatic CPU fallback if the GPU cannot be brought up
- **Batch processing** — point it at a folder and walk away
- **Five output formats** — TXT, SRT, VTT, JSON with timings and quality metrics, and a Markdown meeting protocol
- **Bilingual UI** — English and Russian, switchable in one click
- **3 presets** — Meeting · Long recording · Difficult audio, each chosen from measurements
- **Vocabulary hint** — seed the decoder with attendee names and jargon so it spells them correctly
- **Persistent settings** — restored next launch from `~/.whisper_gui/settings.json`
- **Field validation** — out-of-range numbers turn red and block Start with an explanation
- **Error banner & per-file ETA** — "File 2/5 — ETA 0:42" above the tabs, no need to open Logs
- **Stop is safe** — stopping mid-file writes a `.partial` transcript instead of discarding the work
- **Hotkeys** — `F5` start, `Esc` stop, `Ctrl+O` audio folder, `Ctrl+L` logs, `Ctrl+Q` quit
- **Result preview & quick open** — read the latest transcript in a popup, or jump to the output folder
- **Drag & drop** (optional) — install `windnd` and drop a file or folder onto the audio path
- **WER / CER evaluation** — built-in script to measure transcription quality
- **Headless CLI** — `transcribe_cli.py` for batch jobs and scripting

---

## 🖥️ Requirements

| Component | Minimum |
|-----------|---------|
| OS | Windows 10 / 11 |
| Python | 3.11 or newer (CI tests 3.12) |
| RAM | 8 GB |
| GPU | NVIDIA GPU, CUDA 12 *(recommended)* |
| VRAM | 2 GB for `whisper-medium`, 4 GB+ for `large` |
| Storage | ~1.5 GB per converted model, plus ~2.3 GB for the environment |

> **CPU-only mode works** and is selected automatically when no GPU is available, but expect it to be many times slower on long recordings.

CUDA does **not** need a system-wide toolkit install: the required cuBLAS and cuDNN
libraries come from pip wheels (`requirements-gpu.txt`) and are wired up at startup
by `whisper_engine/cuda_dlls.py`.

---

## ⚡ Quick Start

Four commands from a clean machine to a working transcript. If any of them fails,
[Troubleshooting](#-troubleshooting) covers what people actually hit.

### 1. Get the code

```bash
git clone https://github.com/Anim-2022/Whisper.git
cd Whisper
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

Windows PowerShell uses `.venv\Scripts\Activate.ps1`; on Linux or macOS it is
`source .venv/bin/activate`. Your prompt should now start with `(.venv)`.

Python 3.11 or newer is required — check with `python --version`.

### 3. Install

```bash
pip install -r requirements.txt
```

Add GPU acceleration if you have an NVIDIA card. It is optional; without it
everything still works, just slower:

```bash
pip install -r requirements-gpu.txt
```

### 4. Download a model

```bash
python tools/download_model.py
```

That fetches `whisper-medium` (~1.5 GB), the recommended default, into
`models/ct2/`. It is already in the format the engine uses, so nothing needs to
be converted and torch is never installed.

See the other options with `python tools/download_model.py --list`, or name one
directly: `python tools/download_model.py small`.

> **This is the only step that uses the internet.** Once the model is on disk the
> application never reaches the network again — it loads with `local_files_only`.

### 5. Run

```bash
Start_Whisper.bat
```

or, without the interface:

```bash
python transcribe_cli.py path/to/audio --formats txt,md
```

In the window: pick the folder holding your recordings, pick the language, press
**Start**. Transcripts land in `audio_to_text/`.

<details>
<summary><b>Already have HuggingFace Whisper weights on disk?</b></summary>

Then you can convert them yourself instead of downloading, which keeps the whole
setup offline. This is the only situation that needs torch and transformers:

```bash
pip install -r requirements-convert.txt
python tools/convert_models.py --list     # see what was found
python tools/convert_models.py            # convert everything
```

The tool looks for the HuggingFace cache layout — `models/models--<org>--<name>/snapshots/<hash>/`
— which is what `huggingface-cli download openai/whisper-medium --cache-dir models`
produces. Converted models are written to `models/ct2/<name>/`.

Conversion copies `tokenizer.json` and `preprocessor_config.json` next to the
weights, and refuses to run without them. Both matter: missing the first, the
engine silently reaches out to the network; missing the second, a 128-mel model
such as `large-v3` is decoded as 80 mel bins and quietly produces nonsense.

Afterwards the original `models--*` folders are only needed to convert again, and
can be deleted.

</details>

### The four requirements files

| File | When |
|------|------|
| `requirements.txt` | Always — the runtime |
| `requirements-gpu.txt` | NVIDIA GPU acceleration (cuBLAS + cuDNN) |
| `requirements-convert.txt` | Only to convert your own HuggingFace weights |
| `requirements-dev.txt` | Only to run the tests and linter |

---

## 🩺 Troubleshooting

**`Could not open requirements file: requirements.txt`**
You are not in the project folder. `cd` into the cloned `Whisper` directory —
`dir` (or `ls`) should show `requirements.txt`.

**`No models installed yet` / the model list is empty**
Run `python tools/download_model.py`. The Refresh button next to the model box
rescans `models/ct2/` without restarting the app.

**`python` is not recognised, or reports 3.10 or older**
Install Python 3.11+ from python.org with "Add Python to PATH" ticked, then
recreate the environment with `py -3.12 -m venv .venv`.

**Transcription runs on the CPU although you have an NVIDIA card**
Install `requirements-gpu.txt`. The Logs tab prints which CUDA directories were
registered at startup, and says explicitly when it falls back to CPU.

**`Library cublas64_12.dll is not found`**
Same cause — the CUDA libraries are missing. `pip install -r requirements-gpu.txt`.

**It is very slow**
Check the Logs tab for `cpu` in the model line. On CPU a long recording takes
many times longer; either install the GPU extras or pick a smaller model.

**Out of memory on the GPU**
Lower **Batch size** in the Advanced tab, or set **Compute type** to
`int8_float16`, which roughly halves VRAM use.

---

## 📁 Project Structure

```
Whisper/
├── whisper_engine/          # Engine package
│   ├── engine.py            #   faster-whisper adapter (the only faster_whisper import)
│   ├── config.py            #   TranscriptionConfig
│   ├── models.py            #   local CT2 model discovery and validation
│   ├── audio.py             #   PyAV decoding, format list
│   ├── writers.py           #   TXT / SRT / VTT / JSON / Markdown
│   ├── progress.py          #   the single owner of every status string
│   ├── timestamps.py        #   subtitle timestamp formatting
│   ├── types.py             #   TranscriptResult / TranscriptSegment
│   ├── device.py            #   CUDA probing without torch
│   └── cuda_dlls.py         #   Windows CUDA DLL discovery
├── whisper_core.py          # Batch orchestration over the engine
├── whisper_gui.py           # CustomTkinter GUI (EN/RU bilingual)
├── gui/                     # GUI package: constants, i18n, settings, widgets
├── start_gui.py             # Entry point
├── transcribe_cli.py        # Headless runner
├── tools/download_model.py  # Download a ready-made model (the easy path)
├── tools/convert_models.py  # One-time HF → CTranslate2 conversion
├── evaluate_transcriptions.py  # WER/CER evaluation tool
├── tests/                   # pytest suite (no GPU or weights required)
├── models/ct2/              # ← Converted models go here (not in repo)
├── audio/                   # ← Default input folder (not in repo)
└── audio_to_text/           # ← Default output folder (not in repo)
```

---

## 🏗 Architecture & Logic

### Pipeline Overview

```mermaid
graph TD
    A["User"] -->|"Settings"| B("Whisper GUI")
    B -->|"TranscriptionConfig"| C["WhisperCore"]

    subgraph step1 ["1. Decode"]
        C --> D["PyAV / FFmpeg<br/>any container"]
        D --> E["16 kHz mono<br/>float32"]
    end

    subgraph step2 ["2. Speech detection"]
        E --> F["Silero VAD<br/>(bundled ONNX)"]
        F --> G["Speech regions<br/>+ padding"]
    end

    subgraph step3 ["3. Inference"]
        G --> H{"Mode?"}
        H -->|"Batched"| I["Pack into parallel<br/>30 s windows"]
        H -->|"Sequential"| J["One window<br/>at a time"]
        I --> K["CTranslate2<br/>decode"]
        J --> K
        K --> L{"Degenerate?"}
        L -->|"compression ratio<br/>or logprob"| M["Retry at higher<br/>temperature"]
        M --> K
        L -->|"OK"| N["Segments with<br/>timestamps"]
    end

    subgraph step4 ["4. Output"]
        N --> O["Restore original<br/>timeline"]
        O --> P["TXT · SRT · VTT<br/>JSON · Markdown"]
    end

    P --> A
    N -.->|"live text"| B
```

Timestamps are mapped back to the original timeline after VAD, so progress
reporting and subtitle timings stay correct even though silence was removed
before decoding.

### Batched vs Sequential

| | 🚀 Batched *(default)* | 🎯 Sequential |
| :--- | :--- | :--- |
| **Speed** | Several times faster; packs many 30 s windows into one GPU call | Roughly a quarter of the speed |
| **Segment length** | One segment per 30 s window | Sentence-level |
| **Subtitles** | Cues are ~30 s — fine as a transcript, unusable as subtitles | Proper subtitle timing |
| **Temperature fallback** | Not available | Retries degenerate output at higher temperature |
| **Previous-text context** | Not available | Optional, improves consistency |
| **Hallucination filter** | Not available | Drops text invented over long silences |

The application does not pretend otherwise: selecting a batched mode together
with a sequential-only option raises a warning banner and the option is cleared,
rather than silently having no effect.

### Guards against degenerate output

Long recordings are where Whisper misbehaves — repetition loops, and text
invented over silence. Four independent guards apply:

```mermaid
graph LR
    A["Decoded segment"] --> B{"no_speech_prob<br/>above threshold?"}
    B -->|"yes"| C["Drop as silence"]
    B -->|"no"| D{"compression ratio<br/>too high?"}
    D -->|"yes — repetition loop"| E["Retry hotter"]
    D -->|"no"| F{"avg logprob<br/>too low?"}
    F -->|"yes — low confidence"| E
    F -->|"no"| G{"long silence with<br/>text over it?"}
    G -->|"yes"| C
    G -->|"no"| H["Keep"]
```

---

## 🎛️ Presets

| Preset | Mode | Speed | Best for |
|--------|------|-------|----------|
| **Meeting** *(default)* | Sequential, beam 5 | 16–26× | Everyday use. The most complete text and 3–5 s segments, so one run serves as a transcript *and* as subtitles |
| **Long recording** | Batched, beam 5 | 53–60× | Multi-hour archives. Pays for it with ~30 s segments, unusable as subtitles, and the occasional word lost where two windows meet |
| **Difficult audio** | Sequential, beam 5 | 15–25× | Quiet rooms, distant microphones. Recovers ~4 % more transcript by keeping faint speech the default discards |

Values come from a benchmark sweep rather than intuition. The comments above
`PRESETS` in `gui/constants.py` record what was measured, and — just as usefully —
which knobs were tried and rejected because they changed nothing.

---

## 🧠 Choosing a model

| Model | Notes |
|-------|-------|
| `whisper-medium` *(default)* | The recommended balance. Keeps punctuation and spells English technical terms correctly |
| `whisper-small` | Faster and lighter, less accurate on difficult speech |
| `whisper-large-v3` | Highest quality, heaviest |
| `whisper-large-v3-turbo` | Fastest, good for drafts. Distilled to four decoder layers: it loses punctuation on long speech and tends to transliterate English terms rather than spell them |

> Measured on ~3 h of Russian lecture audio containing English technical terms,
> `medium` in batched mode matched or beat the previous transformers-based engine
> on punctuation density while retaining more English terms — and ran 5.7× faster.
> `turbo` was faster still but produced one test file with no punctuation at all.

---

## ⚙️ Advanced Settings

All advanced parameters are accessible from the **Advanced** tab:

### Compute

| Parameter | Description |
|-----------|-------------|
| Device | `cuda` or `cpu` |
| Compute type | `auto` (float16 on GPU, int8 on CPU), `int8_float16` to halve VRAM, `float32` for debugging |
| Mode | Batched or sequential — see the comparison above |
| Batch size | 30 s windows decoded together. Lower this first on VRAM errors |

### Voice activity detection

| Parameter | Description |
|-----------|-------------|
| VAD threshold | Higher = more aggressive silence cutting |
| Min speech (ms) | Sounds shorter than this are not treated as speech |
| Min silence (ms) | How long a pause must be before speech is split there |
| Speech padding (ms) | Audio kept on both sides so syllables are not clipped |

### Decoding

| Parameter | Description |
|-----------|-------------|
| Beam size | 1 is fastest, 5 is the usual quality choice |
| Temperature fallback | Retry degenerate output at higher temperature *(sequential only)* |
| Previous-text context | Improves consistency, classic cause of repetition loops *(sequential only)* |
| Word timestamps | Per-word timings in the JSON output |
| No-speech threshold | Above this probability a segment is dropped as silence |
| Compression ratio limit | Text compressing better than this is a repetition loop |
| Log-probability threshold | Low-confidence segments are retried |
| Hallucination filter (sec) | Drops text over silences longer than this *(sequential only)* |

---

## 📄 Output Formats

| Format | Contents |
|--------|----------|
| **TXT** | Plain transcript, split into paragraphs on pauses longer than 2 s |
| **SRT** | Standard subtitles, `HH:MM:SS,mmm` |
| **VTT** | WebVTT subtitles, `HH:MM:SS.mmm` |
| **JSON** | Full structured output — see below |
| **Markdown** | Readable meeting protocol: metadata table, then 5-minute sections with timestamps |

The JSON schema (`whisper-unified/transcript`, version 1):

```json
{
  "schema": "whisper-unified/transcript",
  "schema_version": 1,
  "source": { "file": "meeting.m4a", "duration_sec": 10289.0, "duration_after_vad_sec": 8134.2 },
  "model": { "id": "whisper-medium", "engine": "faster-whisper", "engine_version": "1.2.1",
             "device": "cuda", "compute_type": "float16" },
  "language": { "code": "ru", "probability": 0.998, "auto_detected": false },
  "options": { "batched": true, "beam_size": 5, "vad_filter": true, "...": "..." },
  "created_at": "2026-08-16T21:14:05+02:00",
  "stopped_early": false,
  "segments": [
    { "id": 1, "start": 0.0, "end": 4.32, "text": "…",
      "no_speech_prob": 0.02, "avg_logprob": -0.21,
      "compression_ratio": 1.42, "temperature": 0.0 }
  ],
  "text": "full concatenated transcript"
}
```

The per-segment quality metrics are there so a bad run can be diagnosed after the
fact: a high `no_speech_prob` next to confident-looking text is the signature of a
hallucination over silence, and a `compression_ratio` above ~2.4 marks a
repetition loop. `words` appears only when word timestamps are enabled.

---

## 📊 Evaluating Transcription Quality

Compare predicted transcripts against reference texts:

```bash
python evaluate_transcriptions.py --pred-dir audio_to_text/ --ref-dir path/to/reference_texts/
```

Output:

```
File                                     WER      CER
------------------------------------------------------------
interview_01.txt                      5.200%   2.100%
lecture_02.txt                        8.700%   3.400%
------------------------------------------------------------
Average                               6.950%   2.750%
```

> Reference `.txt` files must have **the same filename** as the corresponding prediction files.

---

## 🔧 Supported Audio Formats

`wav` · `mp3` · `flac` · `ogg` · `opus` · `m4a` · `mp4` · `aac` · `wma` · `webm` ·
`mkv` · `mov` · `avi` · `aiff` · `amr` · `3gp`

Decoding goes through PyAV, which bundles FFmpeg — video containers work too, and
the audio track is extracted automatically. No system FFmpeg install is required.

---

## 🧪 Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
ruff check .
```

The test suite runs without a GPU or any model weights: the engine is injected,
so the whole per-file loop is exercised against a stub. CI runs on Linux and
Windows.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](LICENSE) © 2026 Anim-2022
