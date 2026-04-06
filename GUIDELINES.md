# GUIDELINES.md
> **FOR COPILOT — NOT FOR THE HUMAN USER**
> This file contains the complete technical reference for building ClassTranscriber. Every API, version, code example, and architectural decision is documented here. Do not invent APIs. Do not guess versions. Everything you need is in this file.

---

## STACK WITH EXACT VERSIONS

| Library | Version | Role |
|---|---|---|
| Python | 3.11.x | Runtime |
| sounddevice | 0.5.1 | Microphone capture |
| soundfile | 0.12.1 | WAV read/write |
| numpy | 1.26.4 | Audio array processing |
| noisereduce | 3.0.3 | Spectral gating noise reduction |
| faster-whisper | 1.1.0 | Speech-to-text transcription |
| pyannote.audio | 3.3.2 | Speaker diarization |
| torch | 2.3.1+cu121 | GPU backend for pyannote and noisereduce |
| torchaudio | 2.3.1+cu121 | Audio backend for pyannote |
| ollama | 0.4.4 | Python client for local Ollama server |
| python-dotenv | 1.0.1 | Load .env variables |

> ⚠️ **CUDA requirement**: faster-whisper requires CUDA 12 + cuDNN 9. pyannote requires the same. Do not use CUDA 11.

---

## REQUIREMENTS.TXT

```
sounddevice==0.5.1
soundfile==0.12.1
numpy==1.26.4
noisereduce==3.0.3
faster-whisper==1.1.0
pyannote.audio==3.3.2
torch==2.3.1
torchaudio==2.3.1
ollama==0.4.4
python-dotenv==1.0.1
pytest==8.3.3
```

Install PyTorch with CUDA 12.1:
```
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the rest:
```
pip install sounddevice soundfile numpy noisereduce faster-whisper pyannote.audio ollama python-dotenv pytest
```

---

## ENVIRONMENT VARIABLES

**`.env.example`** (copy to `.env` and fill in):
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Load in `main.py`:
```python
from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
```

**Where to get `HF_TOKEN`:**  
Create a free account at https://huggingface.co, go to https://huggingface.co/settings/tokens, create a token with **read** permissions only.  
Then go to https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the model's user conditions (required once, free).

---

## API DOCUMENTATION

---

### 1. sounddevice + soundfile — Audio Capture

**Official docs:** https://python-sounddevice.readthedocs.io/  
**PyPI:** https://pypi.org/project/sounddevice/

**Record from microphone (synchronous):**
```python
import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
DURATION_SECONDS = 7200  # 2 hours
CHANNELS = 1

audio = sd.rec(
    int(DURATION_SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype='float32'
)
sd.wait()  # blocks until recording is done

sf.write("recording.wav", audio, SAMPLE_RATE)
```

**Query available devices:**
```python
import sounddevice as sd
print(sd.query_devices())
device_info = sd.query_devices(kind='input')
print(device_info['name'])
```

**Read/write WAV files:**
```python
import soundfile as sf

# Read
data, samplerate = sf.read("audio.wav", dtype='float32')

# Write
sf.write("audio.wav", data, samplerate=16000)

# Get file info without loading
info = sf.info("audio.wav")
print(info.samplerate, info.channels, info.duration)
```

> **Note:** `sounddevice` does NOT require FFmpeg to be installed. `soundfile` handles WAV natively.

---

### 2. noisereduce — Noise Reduction

**Official docs / GitHub:** https://github.com/timsainb/noisereduce  
**PyPI:** https://pypi.org/project/noisereduce/  
**Current version:** 3.0.3

**Basic usage (non-stationary, recommended for classroom):**
```python
import noisereduce as nr
import soundfile as sf
import numpy as np

data, sr = sf.read("input.wav", dtype='float32')

# Non-stationary: dynamically updates noise threshold over time
# Best for recordings where background noise changes (classroom, AC, etc.)
reduced = nr.reduce_noise(
    y=data,
    sr=sr,
    stationary=False,       # non-stationary mode
    prop_decrease=0.75      # 75% noise reduction — avoids over-suppressing speech
)

# Normalize
reduced = reduced / np.max(np.abs(reduced))

sf.write("output.wav", reduced, sr)
```

**With GPU acceleration (PyTorch):**
```python
import torch
from noisereduce.torchgate import TorchGate as TG

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tg = TG(sr=16000, nonstationary=True).to(device)

# audio_tensor must be shape (batch, samples)
import torch
audio_tensor = torch.tensor(data).unsqueeze(0).to(device)
enhanced = tg(audio_tensor)
enhanced_np = enhanced.squeeze(0).cpu().numpy()
```

> **prop_decrease=0.75** is critical. Using 1.0 will eliminate too much signal, especially for distant or accented speakers. 0.75 is the safe default for classroom use.

---

### 3. pyannote.audio — Speaker Diarization

**Official docs / GitHub:** https://github.com/pyannote/pyannote-audio  
**Model card:** https://huggingface.co/pyannote/speaker-diarization-3.1  
**PyPI:** https://pypi.org/project/pyannote-audio/  
**Version:** 3.3.2 (compatible with speaker-diarization-3.1 model)

**Installation:**
```
pip install pyannote.audio
```

**Full usage:**
```python
from pyannote.audio import Pipeline
import torch

# Load pipeline (downloads model on first run, ~1GB)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

# Send to GPU — REQUIRED for this project
pipeline.to(torch.device("cuda"))

# Run diarization
# Input must be: mono WAV, 16kHz
diarization = pipeline(
    "audio.wav",
    min_speakers=1,
    max_speakers=6
)

# Extract segments
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker  # e.g. "SPEAKER_00", "SPEAKER_01"
    })

# With progress hook (useful for long files)
from pyannote.audio.pipelines.utils.hook import ProgressHook
with ProgressHook() as hook:
    diarization = pipeline("audio.wav", hook=hook)
```

**Important constraints:**
- Input **must** be mono WAV at **16kHz**. Multi-channel audio is automatically downmixed, but it is safer to pre-convert.
- The `use_auth_token` parameter accepts the HuggingFace token string.
- The model requires accepting user conditions at https://huggingface.co/pyannote/speaker-diarization-3.1

**Processing time:** Approximately 2.5% real-time on GPU. For a 2-hour class: ~3 minutes on RTX 4050.

---

### 4. faster-whisper — Transcription

**Official docs / GitHub:** https://github.com/SYSTRAN/faster-whisper  
**PyPI:** https://pypi.org/project/faster-whisper/  
**Version:** 1.1.0

**CUDA requirements:**
- CUDA 12 + cuDNN 9 (latest ctranslate2 versions)
- If you have cuDNN 8 with CUDA 12: `pip install --force-reinstall ctranslate2==4.4.0`

**Basic usage:**
```python
from faster_whisper import WhisperModel

# Initialize once — do NOT re-initialize on every transcription call
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"  # FP16 on RTX 4050
)

# Transcribe
segments, info = model.transcribe(
    "audio.wav",
    beam_size=5,
    language="en",           # Force English — avoids language detection overhead
    vad_filter=True,         # Skip silent segments (speeds up processing)
    word_timestamps=False    # Not needed for this project
)

# IMPORTANT: segments is a GENERATOR — you must iterate or convert to list
# The transcription does not actually run until you consume the generator
segments_list = list(segments)  # This triggers the actual transcription

for segment in segments_list:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

# Log detected language
print(f"Detected language: {info.language} ({info.language_probability:.2%})")
```

**Model sizes and VRAM usage (RTX 4050 = 6GB VRAM):**
| Model | VRAM | Speed (GPU) |
|---|---|---|
| large-v3 | ~3GB FP16 | ~15x real-time on RTX 4050 |
| medium | ~1.5GB | ~25x real-time |
| small | ~600MB | ~50x real-time |

> `large-v3` is recommended for Chinese-accented English. The larger model handles non-native accents significantly better than medium/small.

**Note:** FFmpeg is NOT required — faster-whisper uses PyAV which bundles FFmpeg internally.

---

### 5. Ollama Python Library — LLM Cleanup

**Official docs:** https://github.com/ollama/ollama-python  
**Ollama API docs:** https://github.com/ollama/ollama/blob/main/docs/api.md  
**PyPI:** https://pypi.org/project/ollama/  
**Version:** 0.4.4

**Prerequisites:**
1. Download and install Ollama from https://ollama.com
2. Pull the model: `ollama pull gemma3:4b`
3. Ollama server starts automatically on Windows (or run `ollama serve` manually)
4. Server listens on `http://localhost:11434`

**Model specs — gemma3:4b:**
- Context window: **128,000 tokens**
- VRAM usage: ~3GB on GPU
- A 2-hour class transcript ≈ 15,000–25,000 tokens — fits comfortably in one context window
- Requires Ollama 0.6 or later

**Python usage:**
```python
from ollama import chat

# Simple call
response = chat(
    model='gemma3:4b',
    messages=[
        {
            'role': 'system',
            'content': 'You are a transcript cleaner...'
        },
        {
            'role': 'user',
            'content': 'Text to clean...'
        }
    ]
)
print(response.message.content)
```

**Check if Ollama is running:**
```python
import requests

def is_ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434")
        return r.status_code == 200
    except Exception:
        return False
```

**Pull a model programmatically (if not already pulled):**
```python
import ollama
ollama.pull('gemma3:4b')
```

> **Important:** gemma3:4b can hold the entire 2-hour transcript in one context call (128k tokens). However, the cleaner module uses chunking as a safety measure in case the transcript is unusually long or if the user switches to a smaller model in the future.

---

## CONFIG.PY REFERENCE

```python
# config.py
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "output"

# Recording
SAMPLE_RATE = 16000       # Hz — required by pyannote
CHANNELS = 1              # Mono — required by pyannote
DEFAULT_DURATION = 7200   # 2 hours in seconds

# Diarization
MIN_SPEAKERS = 1
MAX_SPEAKERS = 6

# Transcription
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_LANGUAGE = "en"
WHISPER_BEAM_SIZE = 5

# Noise reduction
NOISE_PROP_DECREASE = 0.75

# LLM cleanup
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_CHUNK_SIZE = 50    # segments per LLM call

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

---

## DATABASE SCHEMA

**Not applicable.** This project does not use a database. All intermediate data is passed as Python objects in memory or as files. The only persisted artifacts are:
- `audio/*.wav` — raw and preprocessed audio files
- `output/*.txt` — final transcripts

---

## DATA FLOW ARCHITECTURE

```
[Microphone]
     │
     ▼ sounddevice.rec()
[audio/{name}_raw.wav]         ← 16kHz mono WAV, up to ~500MB for 2h
     │
     ▼ noisereduce.reduce_noise()
[audio/{name}_clean.wav]       ← noise-reduced, normalized
     │
     ├──────────────────────────────────┐
     ▼                                  ▼
[pyannote Pipeline]            [faster-whisper Model]
     │                                  │
     ▼                                  ▼
[diarization: list[dict]]      [transcription: list[dict]]
  {start, end, speaker}          {start, end, text}
     │                                  │
     └──────────┬───────────────────────┘
                ▼ merger.merge()
        [merged: list[dict]]
          {start, end, speaker, text}
                │
                ▼ cleaner.clean_transcript()
        [cleaned: list[dict]]         ← Ollama gemma3:4b corrects professor's text
          {start, end, speaker, text}
                │
                ▼ exporter.export()
        [output/{name}.txt]           ← final human-readable transcript
```

---

## OUTPUT FORMAT EXAMPLE

```
# ClassTranscriber — 2024-01-15 09:00

[00:00:02] SPEAKER_00: Welcome everyone, today we are going to discuss the fundamentals of distributed computing.

[00:00:15] SPEAKER_01: Professor, can you explain what consistency means in this context?

[00:00:22] SPEAKER_00: Yes, consistency in distributed systems refers to the property that all nodes in the system observe the same data at the same time.

[00:01:05] SPEAKER_00: This is known as linearizability, which is the strongest consistency model available.
```

---

## CODE RULES

1. **Python 3.11 type hints everywhere.** Use `list[dict]`, `Path`, `str | None`, etc.
2. **No `print()` statements.** Use `logging.getLogger(__name__)` in every module.
3. **All file paths via `pathlib.Path`.** Never use string concatenation for paths.
4. **Guard all I/O with try/except** and re-raise with meaningful messages.
5. **No global mutable state** except the `_model` singleton in `transcriber.py`.
6. **All WAV files must be 16kHz mono.** Assert this before passing to pyannote or whisper.
7. **Ollama calls must handle `ConnectionRefusedError`** — if Ollama is not running, raise a clear `RuntimeError("Ollama is not running. Start it with: ollama serve")`.
8. **Tests use `pytest` and `unittest.mock`.** Never call real APIs in unit tests.
9. **`.env` is never committed.** Add it to `.gitignore`. Only `.env.example` is committed.
10. **All output text files use UTF-8 encoding** with `encoding="utf-8"`.

---

## ARCHITECTURAL DECISIONS

### Why faster-whisper instead of openai-whisper?
faster-whisper is a CTranslate2 reimplementation of Whisper that runs up to 4x faster with lower VRAM usage. On the RTX 4050, `large-v3` in FP16 takes ~3GB VRAM and transcribes 2h of audio in approximately 8–12 minutes. The standard `openai-whisper` would take 30–45 minutes for the same task.

### Why large-v3 specifically?
large-v3 is the highest-accuracy Whisper model. For Chinese-accented English, accuracy drops significantly on smaller models. The WER (word error rate) difference between `medium` and `large-v3` for non-native accented English is approximately 15–25%. Given this project's purpose, accuracy is the priority.

### Why pyannote speaker-diarization-3.1 and not 3.0?
Version 3.1 removes the dependency on `onnxruntime` and runs entirely in pure PyTorch. This makes installation simpler on Windows and GPU inference faster. Version 3.0 had known issues on Windows with ONNX session management.

### Why process audio at the end of class, not in real time?
Real-time diarization + transcription would require running both models simultaneously, consuming ~6GB VRAM (pyannote ~1.5GB + whisper large-v3 ~3GB). The RTX 4050 has 6GB total VRAM. Running sequentially after class avoids VRAM contention and eliminates complexity in the recording module.

### Why gemma3:4b over larger models?
gemma3:4b fits comfortably in VRAM after whisper and pyannote are unloaded, runs at fast inference speeds on the RTX 4050, and has a 128k context window that fits an entire 2-hour transcript. The task (grammar correction, punctuation) does not require a 70B model. Testing showed gemma3:4b produces output quality indistinguishable from gemma3:12b for this specific cleanup task.

### Why chunked LLM calls even if 128k fits?
Defense-in-depth. If the user later switches to a model with a smaller context window (e.g., `llama3.2:3b` at 8k), the code will still work. Chunking also allows streaming progress feedback to the user.

### Why mono 16kHz throughout the pipeline?
pyannote.audio's diarization model is trained on 16kHz mono audio. faster-whisper also resamples internally to 16kHz. Keeping all intermediate files at 16kHz mono avoids silent format mismatch bugs and reduces file sizes (a 2h recording at 16kHz mono PCM16 ≈ 115MB vs 460MB at 44.1kHz stereo).

---

## OFFICIAL DOCUMENTATION LINKS

| Component | Link |
|---|---|
| sounddevice | https://python-sounddevice.readthedocs.io/ |
| soundfile | https://python-soundfile.readthedocs.io/ |
| noisereduce | https://github.com/timsainb/noisereduce |
| pyannote.audio | https://github.com/pyannote/pyannote-audio |
| pyannote model card | https://huggingface.co/pyannote/speaker-diarization-3.1 |
| faster-whisper | https://github.com/SYSTRAN/faster-whisper |
| faster-whisper PyPI | https://pypi.org/project/faster-whisper/ |
| Ollama | https://ollama.com |
| Ollama Python library | https://github.com/ollama/ollama-python |
| Ollama API reference | https://github.com/ollama/ollama/blob/main/docs/api.md |
| gemma3:4b model page | https://ollama.com/library/gemma3:4b |
| HuggingFace tokens | https://huggingface.co/settings/tokens |
| PyTorch CUDA install | https://pytorch.org/get-started/locally/ |
| CTranslate2 CUDA notes | https://opennmt.net/CTranslate2/installation.html |
