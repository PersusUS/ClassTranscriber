# INSTRUCTIONS.md
> **FOR COPILOT — NOT FOR THE HUMAN USER**
> Read this file completely before writing any code. Follow every step in order. Do not skip steps. Do not invent solutions not described here. When in doubt, consult GUIDELINES.md.

---

## PROJECT OVERVIEW

**Name:** ClassTranscriber  
**Purpose:** Record classroom audio, separate speakers via diarization, transcribe with Whisper, and produce a clean .txt file per class session corrected by a local LLM.  
**Platform:** Windows 11, Python 3.11, NVIDIA RTX 4050 (CUDA 12), local execution only.  
**User:** Single user, CLI only — no web UI, no authentication, no database.

---

## FOLDER STRUCTURE

Create exactly this structure. Do not add or remove folders.

```
classtranscriber/
├── main.py                  # Entry point CLI
├── config.py                # All constants and user-configurable settings
├── requirements.txt         # Pinned dependencies
├── .env.example             # Template for environment variables
├── README.md                # Setup instructions for the human
│
├── modules/
│   ├── __init__.py
│   ├── recorder.py          # M1 — Audio capture
│   ├── preprocessor.py      # M2 — Noise reduction & normalization
│   ├── diarizer.py          # M3 — Speaker diarization
│   ├── transcriber.py       # M4 — Whisper transcription
│   ├── merger.py            # M5 — Align diarization + transcription segments
│   ├── cleaner.py           # M6 — LLM text cleanup via Ollama
│   └── exporter.py          # M7 — Write final .txt output
│
├── tests/
│   ├── __init__.py
│   ├── test_recorder.py
│   ├── test_preprocessor.py
│   ├── test_diarizer.py
│   ├── test_transcriber.py
│   ├── test_merger.py
│   ├── test_cleaner.py
│   └── test_exporter.py
│
├── audio/                   # Gitignored. Raw recordings saved here.
└── output/                  # Gitignored. Final .txt files saved here.
```

---

## ABSOLUTE RULES FOR COPILOT

1. **Never use `pyaudio`** — use `sounddevice` + `soundfile` only.
2. **Never call OpenAI, Google, Anthropic or any external API** — all processing is local.
3. **Never load the entire audio into RAM at once** — always use chunked or file-based processing.
4. **Never hardcode file paths** — use `pathlib.Path` and values from `config.py`.
5. **pyannote pipeline must always run on GPU** — call `.to(torch.device("cuda"))` after loading.
6. **faster-whisper must always use `device="cuda"` and `compute_type="float16"`**.
7. **Ollama must always be called via the `ollama` Python library**, not via `requests` or `subprocess`.
8. **All audio intermediate files must be mono WAV at 16kHz** — this is required by pyannote.
9. **All functions must have docstrings** and must validate their inputs with clear error messages.
10. **Do not use `print()` for logging** — use Python's `logging` module at appropriate levels.
11. **Each module is independent** — a module must not import from another module except through the interfaces defined in `main.py`.
12. **Do not generate the next module until the current module passes all its tests.**

---

## MODULE SPECIFICATIONS

---

### M1 — `modules/recorder.py`

**Purpose:** Record audio from the default microphone to disk in real time, saving to a WAV file.

**Functions to implement:**

```python
def record(output_path: Path, duration_seconds: int, sample_rate: int = 16000, channels: int = 1) -> Path:
    """
    Records audio from the default system microphone for the given duration.
    Saves to output_path as a 16kHz mono WAV file.
    Logs progress every 10 seconds.
    Returns the path to the saved file.
    Raises FileNotFoundError if the output directory does not exist.
    Raises RuntimeError if no input device is found.
    """
```

**Implementation notes:**
- Use `sounddevice.rec()` with `samplerate=sample_rate, channels=channels, dtype='float32'`.
- Use `sounddevice.wait()` to block until recording completes.
- Use `soundfile.write()` to save the WAV file.
- The output file must always be mono (channels=1) and 16kHz — these are required by downstream modules.
- Log the device name before starting.

**Steps:**
1. Import `sounddevice`, `soundfile`, `numpy`, `pathlib.Path`, `logging`.
2. Validate that `output_path.parent` exists, raise `FileNotFoundError` if not.
3. Query `sounddevice.query_devices(kind='input')` and log the device name.
4. Call `sounddevice.rec()`, then `sounddevice.wait()`.
5. Write to file with `soundfile.write()`.
6. Return `output_path`.

---

**M1 TESTS — `tests/test_recorder.py`**

**Unit tests (automated with pytest):**
- [ ] `test_record_creates_file`: Mock `sounddevice.rec` and `sounddevice.wait`. Assert output WAV file is created at the given path.
- [ ] `test_record_invalid_directory`: Pass a path with a nonexistent parent directory. Assert `FileNotFoundError` is raised.
- [ ] `test_record_returns_path`: Assert the return value equals the given `output_path`.
- [ ] `test_record_correct_samplerate`: Use `soundfile.info()` on the output file to assert `samplerate == 16000`.

**Manual test (run once before committing M1):**
- [ ] Run `python main.py record --duration 10 --output audio/test_m1.wav`
- [ ] Verify `audio/test_m1.wav` exists and plays audio through any media player.
- [ ] Verify file is mono and 16kHz using `python -c "import soundfile as sf; print(sf.info('audio/test_m1.wav'))"`.

---

### M2 — `modules/preprocessor.py`

**Purpose:** Apply noise reduction and volume normalization to a WAV file to improve quality before diarization and transcription.

**Functions to implement:**

```python
def preprocess(input_path: Path, output_path: Path, stationary: bool = False) -> Path:
    """
    Loads a WAV file, applies non-stationary spectral gating noise reduction,
    normalizes the volume, and saves the result to output_path.
    stationary=False uses non-stationary mode (recommended for classroom audio).
    Returns output_path.
    Raises FileNotFoundError if input_path does not exist.
    """
```

**Implementation notes:**
- Use `soundfile.read()` to load audio as float32 numpy array.
- Use `noisereduce.reduce_noise(y=audio, sr=sr, stationary=stationary, prop_decrease=0.75)`.
- After noise reduction, normalize: `audio = audio / np.max(np.abs(audio))`.
- Save with `soundfile.write()`.
- Use `prop_decrease=0.75` — not 1.0, which over-suppresses speech.

**Steps:**
1. Import `soundfile`, `noisereduce`, `numpy`, `pathlib.Path`, `logging`.
2. Validate `input_path.exists()`, raise `FileNotFoundError` if not.
3. Load audio with `soundfile.read(input_path, dtype='float32')`.
4. Apply `noisereduce.reduce_noise()`.
5. Normalize amplitude.
6. Save to `output_path`.
7. Return `output_path`.

---

**M2 TESTS — `tests/test_preprocessor.py`**

**Unit tests:**
- [ ] `test_preprocess_creates_output_file`: Assert output file is created after calling `preprocess()`.
- [ ] `test_preprocess_invalid_input`: Pass a nonexistent input path. Assert `FileNotFoundError` is raised.
- [ ] `test_preprocess_output_normalized`: Load the output file and assert `np.max(np.abs(audio)) <= 1.0`.
- [ ] `test_preprocess_same_samplerate`: Assert input and output have the same sample rate.

**Manual test:**
- [ ] Run `python main.py preprocess --input audio/test_m1.wav --output audio/test_m2.wav`
- [ ] Play `audio/test_m2.wav` and compare to `test_m1.wav` — background hiss should be reduced.

---

### M3 — `modules/diarizer.py`

**Purpose:** Run speaker diarization on a preprocessed WAV file. Returns a list of time segments with speaker labels.

**Functions to implement:**

```python
def diarize(audio_path: Path, hf_token: str, min_speakers: int = 1, max_speakers: int = 6) -> list[dict]:
    """
    Runs pyannote speaker-diarization-3.1 on the given WAV file (must be mono 16kHz).
    Returns a list of dicts: [{"start": float, "end": float, "speaker": str}, ...]
    sorted by start time.
    hf_token: HuggingFace access token (read-only, from .env).
    Raises FileNotFoundError if audio_path does not exist.
    Raises RuntimeError if CUDA is not available.
    """
```

**Implementation notes:**
- Load pipeline with `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)`.
- Send to GPU: `pipeline.to(torch.device("cuda"))`.
- Run: `diarization = pipeline(str(audio_path), min_speakers=min_speakers, max_speakers=max_speakers)`.
- Iterate with `diarization.itertracks(yield_label=True)` to extract segments.
- Speaker labels will be strings like `"SPEAKER_00"`, `"SPEAKER_01"`, etc.

**Steps:**
1. Import `pyannote.audio.Pipeline`, `torch`, `pathlib.Path`, `logging`.
2. Assert `torch.cuda.is_available()`, raise `RuntimeError` if not.
3. Load and send pipeline to CUDA.
4. Run diarization.
5. Build and return sorted list of segment dicts.

---

**M3 TESTS — `tests/test_diarizer.py`**

**Unit tests:**
- [ ] `test_diarize_returns_list`: Mock the pipeline. Assert return type is `list`.
- [ ] `test_diarize_segment_keys`: Assert each dict in the result contains keys `"start"`, `"end"`, `"speaker"`.
- [ ] `test_diarize_sorted_by_start`: Assert segments are sorted by `"start"` ascending.
- [ ] `test_diarize_invalid_path`: Assert `FileNotFoundError` is raised for missing file.

**Manual test:**
- [ ] Run `python main.py diarize --input audio/test_m2.wav`
- [ ] Verify the console prints a list of segments like: `[0.5s -> 12.3s] SPEAKER_00`
- [ ] Verify at least 2 speakers are detected if a multi-speaker audio is used.

---

### M4 — `modules/transcriber.py`

**Purpose:** Transcribe a WAV file using faster-whisper large-v3 on GPU. Returns segments with timestamps.

**Functions to implement:**

```python
def transcribe(audio_path: Path, language: str = "en") -> list[dict]:
    """
    Transcribes the given audio file using faster-whisper large-v3 on CUDA.
    Returns a list of dicts: [{"start": float, "end": float, "text": str}, ...]
    language: ISO 639-1 code. Default "en" (English).
    Raises FileNotFoundError if audio_path does not exist.
    """
```

**Implementation notes:**
- Model: `WhisperModel("large-v3", device="cuda", compute_type="float16")`.
- Call: `segments, info = model.transcribe(str(audio_path), beam_size=5, language=language, vad_filter=True, word_timestamps=False)`.
- `segments` is a generator — consume it with `list(segments)`.
- Log detected language and probability.
- Do NOT reinitialize the model on every call — use a module-level singleton or pass the model in.

**Steps:**
1. Import `faster_whisper.WhisperModel`, `pathlib.Path`, `logging`.
2. Define `_model = None` at module level for lazy loading.
3. In `transcribe()`, initialize `_model` if it is `None`.
4. Validate `audio_path.exists()`.
5. Run transcription, consume generator, return list of dicts.

---

**M4 TESTS — `tests/test_transcriber.py`**

**Unit tests:**
- [ ] `test_transcribe_returns_list`: Mock `WhisperModel`. Assert return type is `list`.
- [ ] `test_transcribe_segment_keys`: Assert each dict contains `"start"`, `"end"`, `"text"`.
- [ ] `test_transcribe_invalid_path`: Assert `FileNotFoundError` for missing file.
- [ ] `test_model_singleton`: Call `transcribe()` twice. Assert `WhisperModel` constructor is called only once (use mock).

**Manual test:**
- [ ] Run `python main.py transcribe --input audio/test_m2.wav`
- [ ] Verify raw transcript is printed to console with timestamps.
- [ ] Verify English is detected with high probability (>0.9 for English classroom audio).

---

### M5 — `modules/merger.py`

**Purpose:** Align diarization segments (who spoke when) with transcription segments (what was said when) to produce speaker-attributed text.

**Functions to implement:**

```python
def merge(diarization: list[dict], transcription: list[dict]) -> list[dict]:
    """
    Merges diarization and transcription segments by time overlap.
    For each transcription segment, assigns the speaker whose diarization segment
    has the maximum overlap with that transcription segment's time window.
    Returns a list of dicts: [{"start": float, "end": float, "speaker": str, "text": str}, ...]
    sorted by start time.
    If no diarization segment overlaps a transcription segment, speaker is "UNKNOWN".
    """
```

**Implementation notes:**
- For each transcription segment `[ts, te]`, find all diarization segments that overlap.
- Overlap = `min(te, de) - max(ts, ds)` where `[ds, de]` is the diarization segment.
- Assign the speaker with the maximum positive overlap.
- If no overlap found, assign speaker `"UNKNOWN"`.
- This function is pure Python — no external libraries needed beyond standard library.

**Steps:**
1. Validate that both inputs are non-empty lists.
2. For each transcription segment, iterate diarization to find max overlap.
3. Build output dict with keys `start`, `end`, `speaker`, `text`.
4. Return sorted by `start`.

---

**M5 TESTS — `tests/test_merger.py`**

**Unit tests:**
- [ ] `test_merge_basic_overlap`: Simple 2-segment case. Assert speaker is correctly assigned.
- [ ] `test_merge_no_overlap`: Transcription segment with no overlapping diarization. Assert speaker is `"UNKNOWN"`.
- [ ] `test_merge_output_keys`: Assert each dict contains `"start"`, `"end"`, `"speaker"`, `"text"`.
- [ ] `test_merge_sorted_output`: Assert output is sorted by `"start"`.
- [ ] `test_merge_empty_inputs`: Assert raises `ValueError` for empty input lists.

**Manual test:**
- [ ] No manual test needed. This module is fully testable via unit tests.

---

### M6 — `modules/cleaner.py`

**Purpose:** Use Ollama + gemma3:4b to clean transcription text: fix grammar, punctuation, remove filler words, and make the text coherent.

**Functions to implement:**

```python
def clean_transcript(segments: list[dict], model: str = "gemma3:4b", chunk_size: int = 50) -> list[dict]:
    """
    Takes merged segments (with speaker labels) and cleans the text of the professor's segments
    using Ollama. Non-professor segments are passed through unchanged.
    Processes segments in chunks of chunk_size to stay within context limits.
    Returns the same list structure with cleaned "text" fields.
    model: Ollama model name.
    chunk_size: number of segments per LLM call.
    """

def set_professor_speaker(speaker_label: str) -> None:
    """
    Sets the module-level variable that identifies which speaker is the professor.
    Must be called before clean_transcript().
    """
```

**Implementation notes:**
- Use `from ollama import chat` (the official ollama Python library).
- The system prompt must instruct the model to: fix broken English grammar from Chinese-accented speech, add punctuation, remove filler words (uh, um, like), preserve technical/academic terminology, return only the corrected text with no preamble.
- Send chunks of text as a single user message: join the segment texts with `\n`.
- After cleanup, re-split the response back into individual segments by line.
- If the LLM response has fewer lines than input segments, pad with the original text.
- The `ollama` service must be running locally on port 11434.

**System prompt to use (exact):**
```
You are a transcript cleaner for academic lectures. The speaker has a strong Chinese accent and speaks non-native English.
Your task: fix grammar, add correct punctuation, remove filler words (uh, um, er, like, you know), 
correct obvious transcription errors, and make the text read naturally.
Preserve all technical and academic terminology exactly as-is.
Respond ONLY with the corrected text, one line per input line. Do not add explanations or numbering.
```

**Steps:**
1. Import `ollama`, `logging`, `pathlib`.
2. Define `_professor_speaker: str = None` at module level.
3. Implement `set_professor_speaker()`.
4. In `clean_transcript()`, split `segments` into chunks of `chunk_size`.
5. For each chunk, filter to professor segments only, build prompt, call Ollama.
6. Replace `text` fields in-place with cleaned versions.
7. Return updated segments list.

---

**M6 TESTS — `tests/test_cleaner.py`**

**Unit tests:**
- [ ] `test_clean_calls_ollama`: Mock `ollama.chat`. Assert it is called at least once.
- [ ] `test_clean_returns_same_structure`: Assert output has the same keys as input.
- [ ] `test_clean_non_professor_unchanged`: Segments from non-professor speakers must not be modified.
- [ ] `test_clean_chunking`: With `chunk_size=2` and 5 segments, assert Ollama is called 3 times (ceil(5/2)).

**Manual test:**
- [ ] Ensure `ollama serve` is running and `gemma3:4b` is pulled.
- [ ] Run `python main.py clean --input output/test_merged.json`
- [ ] Inspect the output — grammar and punctuation should be improved.

---

### M7 — `modules/exporter.py`

**Purpose:** Write the final cleaned transcript to a plaintext .txt file.

**Functions to implement:**

```python
def export(segments: list[dict], output_path: Path, include_timestamps: bool = True) -> Path:
    """
    Writes the transcript to a .txt file.
    Format per line: [HH:MM:SS] SPEAKER_00: text
    If include_timestamps is False, omits the timestamp prefix.
    Returns output_path.
    Raises ValueError if segments is empty.
    """
```

**Implementation notes:**
- Use `datetime.timedelta` to format timestamps from float seconds to `HH:MM:SS`.
- Write UTF-8 encoded file.
- Add a header line: `# ClassTranscriber — {datetime.now().strftime("%Y-%m-%d %H:%M")}`
- Add a blank line between different speaker turns (when speaker changes).

**Steps:**
1. Validate `segments` is not empty.
2. Open `output_path` with `open(..., "w", encoding="utf-8")`.
3. Write header.
4. Iterate segments, format and write each line.
5. Add blank line on speaker change.
6. Return `output_path`.

---

**M7 TESTS — `tests/test_exporter.py`**

**Unit tests:**
- [ ] `test_export_creates_file`: Assert output file is created.
- [ ] `test_export_contains_speaker_labels`: Assert output file text contains `"SPEAKER_"`.
- [ ] `test_export_timestamp_format`: Assert timestamps match pattern `\[\d{2}:\d{2}:\d{2}\]`.
- [ ] `test_export_empty_segments`: Assert `ValueError` is raised for empty input.
- [ ] `test_export_utf8`: Write a segment with special characters and assert they are preserved.

**Manual test:**
- [ ] Inspect the final `.txt` file in `output/` — it must be human-readable without any code artifacts.

---

## MAIN ENTRY POINT — `main.py`

Implement a CLI using `argparse` with the following subcommands:

```
python main.py record --duration 7200 --output audio/class_2024_01_15.wav
python main.py preprocess --input audio/class_2024_01_15.wav --output audio/class_2024_01_15_clean.wav
python main.py diarize --input audio/class_2024_01_15_clean.wav
python main.py transcribe --input audio/class_2024_01_15_clean.wav
python main.py clean --input output/merged.json --professor SPEAKER_00
python main.py run --duration 7200 --name class_2024_01_15
```

The `run` subcommand executes the full pipeline end-to-end:
1. Record to `audio/{name}_raw.wav`
2. Preprocess to `audio/{name}_clean.wav`
3. Diarize
4. Transcribe
5. Merge
6. Clean (ask user which speaker is the professor via interactive prompt)
7. Export to `output/{name}.txt`

Load `HF_TOKEN` from `.env` using `python-dotenv`.

---

## END-TO-END TEST (run after all modules pass their tests)

**E2E Checklist:**

- [ ] Record a 5-minute audio with at least 2 people speaking (use yourself + a video on YouTube playing in the background, or record with a classmate).
- [ ] Run the full pipeline: `python main.py run --duration 300 --name e2e_test`
- [ ] Verify `audio/e2e_test_raw.wav` exists and is ~5 minutes.
- [ ] Verify `audio/e2e_test_clean.wav` exists and sounds cleaner than the raw.
- [ ] Verify diarization detected at least 2 speakers.
- [ ] Verify transcription produced text in English.
- [ ] Verify merged output assigns text to speaker labels.
- [ ] Verify `output/e2e_test.txt` exists, has a header, has speaker labels, has timestamps, and the text is grammatically corrected.
- [ ] Verify the total runtime (excluding recording) is under 10 minutes for 2h of audio on RTX 4050.

---

## COMMIT STRATEGY

Commit after each module passes ALL its tests:
- `git commit -m "M1: recorder — all tests pass"`
- `git commit -m "M2: preprocessor — all tests pass"`
- `git commit -m "M3: diarizer — all tests pass"`
- `git commit -m "M4: transcriber — all tests pass"`
- `git commit -m "M5: merger — all tests pass"`
- `git commit -m "M6: cleaner — all tests pass"`
- `git commit -m "M7: exporter — all tests pass"`
- `git commit -m "main: CLI and E2E pipeline complete"`
