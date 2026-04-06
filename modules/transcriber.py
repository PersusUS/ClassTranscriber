"""M4 — Whisper transcription module.

Transcribes a WAV file using faster-whisper large-v3 on GPU.
Returns segments with timestamps.
"""

import logging
from pathlib import Path

from faster_whisper import WhisperModel

from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_BEAM_SIZE

logger = logging.getLogger(__name__)

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Lazily initializes and returns the WhisperModel singleton.

    Returns:
        The shared WhisperModel instance.
    """
    global _model
    if _model is None:
        # Try configured settings first, then fall back on OOM
        fallbacks = [
            (WHISPER_DEVICE, WHISPER_COMPUTE_TYPE),
            (WHISPER_DEVICE, "int8") if WHISPER_DEVICE == "cuda" else None,
            ("cpu", "int8"),
        ]
        for fb in fallbacks:
            if fb is None:
                continue
            device, compute = fb
            logger.info(
                "Loading Whisper model '%s' on %s (%s)",
                WHISPER_MODEL,
                device,
                compute,
            )
            try:
                _model = WhisperModel(
                    WHISPER_MODEL,
                    device=device,
                    compute_type=compute,
                )
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning(
                        "Failed to load on %s (%s): %s — trying next fallback",
                        device,
                        compute,
                        exc,
                    )
                    continue
                raise
        if _model is None:
            raise RuntimeError("Could not load Whisper model with any configuration")
    return _model


def transcribe(audio_path: Path, language: str = "en") -> list[dict]:
    """Transcribes the given audio file using faster-whisper large-v3 on CUDA.

    Args:
        audio_path: Path to the WAV file to transcribe.
        language: ISO 639-1 language code. Default "en" (English).

    Returns:
        A list of dicts: [{"start": float, "end": float, "text": str}, ...]

    Raises:
        FileNotFoundError: If audio_path does not exist.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    model = _get_model()

    logger.info("Transcribing %s (language=%s)", audio_path, language)
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=WHISPER_BEAM_SIZE,
        language=language,
        vad_filter=True,
        word_timestamps=False,
    )

    # segments is a generator — consume it to trigger actual transcription
    segments_list = list(segments)

    logger.info(
        "Detected language: %s (%.2f%% probability)",
        info.language,
        info.language_probability * 100,
    )
    logger.info("Transcription complete: %d segments", len(segments_list))

    return [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        }
        for seg in segments_list
    ]
