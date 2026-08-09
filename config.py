# config.py
"""Central configuration for ClassTranscriber.

Everything tunable lives here. The most important concept is the
*profile*: a bundle of model sizes and decoding settings that trades
accuracy for CPU/RAM usage, so the same code runs on a laptop without a
GPU and on a machine with CUDA.

Profiles
--------
low       Smallest footprint. Runs comfortably on 4 CPU cores / 8 GB RAM.
balanced  Default. large-v3-turbo in int8 — near large-v3 quality at a
          fraction of the cost.
quality   large-v3 with beam search. Worth it only with a GPU.

Any value can be overridden with an environment variable of the same
name (see `_env_*` helpers), which makes it easy to experiment without
editing this file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "output"
SESSIONS_DIR = BASE_DIR / "sessions"

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000       # Hz — required by pyannote and Whisper
CHANNELS = 1              # Mono — required by pyannote
DEFAULT_DURATION = 7200   # 2 hours in seconds
RECORD_BLOCK_SECONDS = 1.0   # Size of each chunk streamed to disk
RECORD_SUBTYPE = "PCM_16"    # 16-bit WAV: half the size of float32, no quality loss here
LEVEL_LOG_INTERVAL = 30      # Seconds between input-level reports while recording

# ---------------------------------------------------------------------------
# Preprocessing (noise robustness)
# ---------------------------------------------------------------------------

HIGHPASS_HZ = 80.0            # Cuts HVAC rumble, projector hum, desk thumps
NOISE_PROP_DECREASE = 0.75    # How aggressively spectral gating attenuates noise
NOISE_PROFILE_SECONDS = 2.0   # Amount of "quietest audio" used as the noise fingerprint
PREPROCESS_BLOCK_SECONDS = 30.0   # Streaming block size — bounds RAM on 2 h files
PREPROCESS_OVERLAP_SECONDS = 0.5  # Crossfaded overlap between blocks

# Target loudness. RMS-based (not peak-based): in a classroom the peak is
# usually a chair scraping, so peak normalisation leaves the professor quiet.
TARGET_DBFS = -20.0
PEAK_CEILING = 0.97       # Safety ceiling to avoid clipping after gain
MAX_GAIN_DB = 25.0        # Never amplify more than this (would just boost noise)

# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

LANGUAGE = os.getenv("CT_LANGUAGE", "es")

# Biases Whisper towards correct punctuation and academic register.
# A well-punctuated prompt makes Whisper punctuate its output too.
INITIAL_PROMPTS = {
    "es": (
        "Transcripción de una clase universitaria en español. "
        "El profesor explica conceptos técnicos, define términos y responde "
        "preguntas de los alumnos. Usa puntuación correcta, mayúsculas y tildes."
    ),
    "en": (
        "Transcript of a university lecture in English. The professor explains "
        "technical concepts, defines terms and answers student questions. "
        "Use correct punctuation and capitalisation."
    ),
}

# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

WHISPER_LANGUAGE = LANGUAGE      # Kept for backwards compatibility
WHISPER_BEAM_SIZE = 5

# Voice activity detection. Aggressive settings pay off in noisy rooms:
# Whisper hallucinates confidently when fed pure background noise.
VAD_FILTER = True
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 200

# Decoding guards against noise-driven hallucination.
NO_SPEECH_THRESHOLD = 0.6
LOG_PROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CONDITION_ON_PREVIOUS_TEXT = False   # Prevents runaway repetition loops
HALLUCINATION_SILENCE_THRESHOLD = 2.0

# Word timestamps let the merger attribute individual words to speakers,
# which is what keeps student questions out of the professor's transcript.
WORD_TIMESTAMPS = True

# Segments this weak are dropped outright.
MIN_SEGMENT_LOGPROB = -1.2
MAX_SEGMENT_NO_SPEECH_PROB = 0.85

# Phrases Whisper famously invents over silence or noise, especially in
# Spanish (they come from subtitled video in its training data).
HALLUCINATION_PATTERNS = (
    "subtítulos realizados por la comunidad de amara.org",
    "subtítulos por la comunidad de amara.org",
    "subtitulado por la comunidad de amara.org",
    "más información en www.",
    "amara.org",
    "gracias por ver el video",
    "gracias por ver el vídeo",
    "suscríbete al canal",
    "no olvides suscribirte",
    "subscribe to my channel",
    "thanks for watching",
)

# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

DIARIZATION_MODEL = os.getenv("CT_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
MIN_SPEAKERS = 1
MAX_SPEAKERS = 6

# Minimum share of speaking time for the top speaker to be auto-accepted
# as the professor without asking.
PROFESSOR_CONFIDENCE_SHARE = 0.55

# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

TURN_MAX_GAP_SECONDS = 1.5    # Same-speaker segments closer than this become one turn
TURN_MAX_SECONDS = 60.0       # ...but never build a turn longer than this
MIN_WORD_SEGMENT_SECONDS = 0.2

# ---------------------------------------------------------------------------
# LLM cleanup
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHUNK_SIZE = 40        # Max segments per LLM call
OLLAMA_CHUNK_CHARS = 3500     # ...and max characters, whichever comes first
OLLAMA_NUM_CTX = 8192
OLLAMA_TEMPERATURE = 0.0
OLLAMA_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("CT_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    """A resource/accuracy trade-off preset."""

    name: str
    whisper_model: str
    beam_size: int
    cpu_compute_type: str
    gpu_compute_type: str
    ollama_model: str
    denoise: bool = True
    description: str = ""


PROFILES: dict[str, Profile] = {
    "low": Profile(
        name="low",
        whisper_model="small",
        beam_size=1,
        cpu_compute_type="int8",
        gpu_compute_type="int8_float16",
        ollama_model="qwen2.5:3b",
        denoise=True,
        description="Minimum footprint: ~1 GB RAM, usable on 4 CPU cores.",
    ),
    "balanced": Profile(
        name="balanced",
        whisper_model="large-v3-turbo",
        beam_size=2,
        cpu_compute_type="int8",
        gpu_compute_type="float16",
        ollama_model="gemma3:4b",
        denoise=True,
        description="Default: near large-v3 accuracy at a fraction of the cost.",
    ),
    "quality": Profile(
        name="quality",
        whisper_model="large-v3",
        beam_size=WHISPER_BEAM_SIZE,
        cpu_compute_type="int8",
        gpu_compute_type="float16",
        ollama_model="gemma3:4b",
        denoise=True,
        description="Best accuracy. Recommended only with a CUDA GPU.",
    ),
}

DEFAULT_PROFILE = os.getenv("CT_PROFILE", "auto")


@dataclass
class Settings:
    """Fully resolved runtime settings, produced by `resolve_settings()`."""

    profile: Profile
    device: str                 # "cuda" or "cpu"
    compute_type: str
    language: str
    cpu_threads: int
    whisper_model: str
    beam_size: int
    ollama_model: str
    denoise: bool
    extra: dict = field(default_factory=dict)

    @property
    def initial_prompt(self) -> str | None:
        return INITIAL_PROMPTS.get(self.language)

    def describe(self) -> str:
        return (
            f"profile={self.profile.name} device={self.device} "
            f"compute={self.compute_type} model={self.whisper_model} "
            f"beam={self.beam_size} lang={self.language} threads={self.cpu_threads}"
        )


def cuda_available() -> bool:
    """Reports whether a usable CUDA device is present.

    Imports torch lazily so that the CLI stays fast and so that machines
    without torch installed still get a sensible answer instead of an
    ImportError.
    """
    try:
        import torch
    except Exception:       # pragma: no cover - depends on the install
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:       # pragma: no cover - driver problems
        return False


def gpu_vram_gb() -> float:
    """Returns the total VRAM of the first CUDA device in GB, or 0.0."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:       # pragma: no cover - depends on the install
        return 0.0


def default_cpu_threads() -> int:
    """Picks a CPU thread count that leaves the laptop usable.

    Uses all cores minus one, capped at 8: beyond that CTranslate2 gains
    little and the fans become the dominant cost.
    """
    cores = os.cpu_count() or 4
    return max(1, min(8, cores - 1))


def resolve_profile(name: str | None = None) -> Profile:
    """Resolves a profile name, honouring "auto" and unknown values.

    "auto" picks `quality` on a GPU with at least 6 GB of VRAM,
    `balanced` on any other CUDA device, and `balanced` on CPU-only
    machines — `low` stays an explicit opt-in for when you want the
    laptop to stay cool.
    """
    name = (name or DEFAULT_PROFILE or "auto").lower()

    if name == "auto":
        if cuda_available():
            return PROFILES["quality"] if gpu_vram_gb() >= 6.0 else PROFILES["balanced"]
        return PROFILES["balanced"]

    if name not in PROFILES:
        raise ValueError(
            f"Unknown profile '{name}'. Choose one of: {', '.join(PROFILES)} or 'auto'."
        )
    return PROFILES[name]


def resolve_settings(
    profile: str | None = None,
    language: str | None = None,
    device: str | None = None,
    cpu_threads: int | None = None,
    whisper_model: str | None = None,
    ollama_model: str | None = None,
    denoise: bool | None = None,
) -> Settings:
    """Builds the concrete settings used by every stage of the pipeline.

    Args:
        profile: Profile name, or "auto" to detect from the hardware.
        language: ISO 639-1 code. Defaults to `LANGUAGE` ("es").
        device: "cuda", "cpu" or "auto". Falls back to CPU when CUDA is
            unavailable instead of failing.
        cpu_threads: Threads for CPU inference. Defaults to cores - 1.
        whisper_model: Overrides the profile's Whisper model.
        ollama_model: Overrides the profile's cleanup model.
        denoise: Overrides the profile's denoising flag.

    Returns:
        A fully populated `Settings` instance.
    """
    prof = resolve_profile(profile)

    requested = (device or "auto").lower()
    if requested == "auto":
        resolved_device = "cuda" if cuda_available() else "cpu"
    elif requested == "cuda" and not cuda_available():
        resolved_device = "cpu"
    else:
        resolved_device = requested

    compute_type = (
        prof.gpu_compute_type if resolved_device == "cuda" else prof.cpu_compute_type
    )

    return Settings(
        profile=prof,
        device=resolved_device,
        compute_type=compute_type,
        language=(language or LANGUAGE).lower(),
        cpu_threads=cpu_threads or default_cpu_threads(),
        whisper_model=whisper_model or prof.whisper_model,
        beam_size=prof.beam_size,
        ollama_model=ollama_model or prof.ollama_model,
        denoise=prof.denoise if denoise is None else denoise,
    )
