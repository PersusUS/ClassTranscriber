# ClassTranscriber

Record classroom audio, separate speakers via diarization, transcribe with Whisper, and produce a clean `.txt` transcript corrected by a local LLM.

Runs entirely locally — no cloud APIs, no data leaves your machine.

## Prerequisites

- **Windows 11**
- **Python 3.11**
- **NVIDIA GPU with CUDA 12** (tested on RTX 4050)
- **Ollama** installed — download from https://ollama.com
- **HuggingFace account** (free) — for the speaker diarization model

## Installation

### 1. Install PyTorch with CUDA (must be first)

```bash
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up HuggingFace token

1. Create a free account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens and create a token with **read** permissions
3. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1 (required once, free)
4. Copy the example env file and add your token:

```bash
copy .env.example .env
```

Edit `.env` and set:

```
HF_TOKEN=hf_your_token_here
```

### 4. Set up Ollama

1. Install Ollama from https://ollama.com
2. Pull the cleanup model:

```bash
ollama pull gemma3:4b
```

3. Start the Ollama server (keep it running in a separate terminal):

```bash
ollama serve
```

## Usage

### Full pipeline (recommended)

Records audio, preprocesses, diarizes speakers, transcribes, cleans with LLM, and exports a `.txt` file:

```bash
python main.py run --duration 7200 --name lecture_01
```

- `--duration` — recording length in seconds (default: 7200 = 2 hours)
- `--name` — session name, used for output filenames

During the pipeline you will be asked to identify which speaker is the professor. The LLM cleanup is applied only to the professor's speech.

Output is saved to `output/<name>.txt`.

### Individual commands

Each pipeline step can be run separately:

#### Record audio

```bash
python main.py record --duration 3600 --output audio/lecture.wav
```

- `--duration` — recording length in seconds (default: 7200)
- `--output` — output WAV file path (required)

#### Preprocess (noise reduction + normalization)

```bash
python main.py preprocess --input audio/lecture.wav --output audio/lecture_clean.wav
```

- `--input` — input WAV file (required)
- `--output` — output WAV file (required)

#### Diarize (speaker separation)

```bash
python main.py diarize --input audio/lecture_clean.wav
```

- `--input` — input WAV file (required)

#### Transcribe (speech-to-text)

```bash
python main.py transcribe --input audio/lecture_clean.wav
```

- `--input` — input WAV file (required)

#### Clean (LLM transcript correction)

```bash
python main.py clean --input merged_segments.json --professor SPEAKER_00
```

- `--input` — JSON file with merged segments (required)
- `--professor` — speaker label of the professor (required)

## Running tests

```bash
pytest
```

## Project structure

```
ClassTranscriber/
├── main.py              # CLI entry point
├── config.py            # All constants and settings
├── requirements.txt     # Pinned dependencies
├── .env.example         # Template for environment variables
├── modules/
│   ├── recorder.py      # M1 — Audio capture
│   ├── preprocessor.py  # M2 — Noise reduction & normalization
│   ├── diarizer.py      # M3 — Speaker diarization
│   ├── transcriber.py   # M4 — Whisper transcription
│   ├── merger.py        # M5 — Align diarization + transcription
│   ├── cleaner.py       # M6 — LLM text cleanup via Ollama
│   └── exporter.py      # M7 — Write final .txt output
├── tests/               # Unit tests
├── audio/               # Raw and processed recordings (gitignored)
└── output/              # Final transcripts (gitignored)
```
