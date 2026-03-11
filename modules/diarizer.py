"""M3 — Speaker diarization module.

Runs pyannote speaker-diarization-3.1 on a preprocessed WAV file
and returns a list of time segments with speaker labels.
"""

import logging
from pathlib import Path

import torch
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)


def diarize(
    audio_path: Path,
    hf_token: str,
    min_speakers: int = 1,
    max_speakers: int = 6,
) -> list[dict]:
    """Runs pyannote speaker-diarization-3.1 on the given WAV file.

    The input file must be mono 16 kHz WAV.

    Args:
        audio_path: Path to the audio file to diarize.
        hf_token: HuggingFace access token (read-only, from .env).
        min_speakers: Minimum expected number of speakers.
        max_speakers: Maximum expected number of speakers.

    Returns:
        A list of dicts sorted by start time:
        [{"start": float, "end": float, "speaker": str}, ...]

    Raises:
        FileNotFoundError: If audio_path does not exist.
        RuntimeError: If CUDA is not available.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. A CUDA-capable GPU is required for diarization."
        )

    logger.info("Loading pyannote speaker-diarization-3.1 pipeline")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(torch.device("cuda"))

    logger.info(
        "Running diarization on %s (min_speakers=%d, max_speakers=%d)",
        audio_path,
        min_speakers,
        max_speakers,
    )
    diarization = pipeline(
        str(audio_path),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    segments.sort(key=lambda s: s["start"])

    logger.info("Diarization complete: %d segments found", len(segments))
    return segments
