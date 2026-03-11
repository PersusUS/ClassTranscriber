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
