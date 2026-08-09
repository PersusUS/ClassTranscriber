"""M3 — Speaker diarization module.

Runs pyannote's speaker-diarization pipeline to find out *who* spoke
*when*. Unlike the original version this does not require CUDA: the
pipeline runs on CPU too, roughly at real time on a modern laptop, which
is what makes the whole tool usable without a GPU.
"""

import logging
import os
import time
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_pipelines: dict[tuple, object] = {}

_TOKEN_HELP = (
    "Diarization needs the pyannote model, which is gated on HuggingFace.\n"
    "  1. Create a free account at https://huggingface.co\n"
    "  2. Accept the licence at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
    "     (and at https://huggingface.co/pyannote/segmentation-3.0)\n"
    "  3. Create a read token at https://huggingface.co/settings/tokens\n"
    "  4. Put it in .env as HF_TOKEN=hf_...\n"
    "Once the model is downloaded it stays cached and works offline."
)


def _resolve_device(device: str | None) -> str:
    """Picks the torch device, falling back to CPU when CUDA is missing."""
    requested = (device or "auto").lower()
    if requested == "auto":
        return "cuda" if config.cuda_available() else "cpu"
    if requested == "cuda" and not config.cuda_available():
        logger.warning("CUDA requested but unavailable — diarizing on CPU")
        return "cpu"
    return requested


def _load_pipeline(model: str, hf_token: str | None, device: str):
    """Loads and caches the diarization pipeline on the given device."""
    import torch
    from pyannote.audio import Pipeline

    key = (model, device)
    if key in _pipelines:
        return _pipelines[key]

    logger.info("Loading diarization pipeline '%s' on %s", model, device)
    started = time.monotonic()
    try:
        pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)
    except Exception as exc:      # noqa: BLE001 - surfaces as an actionable message
        raise RuntimeError(f"Could not load '{model}'.\n{_TOKEN_HELP}") from exc

    if pipeline is None:
        # pyannote returns None instead of raising when the licence has not
        # been accepted or the token lacks access.
        raise RuntimeError(f"Could not load '{model}'.\n{_TOKEN_HELP}")

    pipeline.to(torch.device(device))

    # Larger batches help on GPU; on CPU they only inflate peak memory.
    if device == "cuda":
        for attribute, value in (("segmentation_batch_size", 32), ("embedding_batch_size", 32)):
            if hasattr(pipeline, attribute):
                setattr(pipeline, attribute, value)

    logger.info("Diarization pipeline ready in %.1f s", time.monotonic() - started)
    _pipelines[key] = pipeline
    return pipeline


def diarize(
    audio_path: Path,
    hf_token: str | None = None,
    min_speakers: int = config.MIN_SPEAKERS,
    max_speakers: int = config.MAX_SPEAKERS,
    num_speakers: int | None = None,
    device: str | None = None,
    model: str = config.DIARIZATION_MODEL,
) -> list[dict]:
    """Splits an audio file into speaker-labelled time segments.

    The input should be mono 16 kHz WAV (what `preprocess` produces).

    Args:
        audio_path: Path to the audio file to diarize.
        hf_token: HuggingFace read token. Falls back to `$HF_TOKEN`, then
            to the local model cache for offline use.
        min_speakers: Lower bound on the number of speakers.
        max_speakers: Upper bound on the number of speakers.
        num_speakers: Exact speaker count, when known. Constraining this
            markedly improves accuracy — in a lecture where only the
            professor and two students talk, pass 3.
        device: "cuda", "cpu" or "auto".
        model: HuggingFace model id of the pipeline.

    Returns:
        A list of dicts sorted by start time:
        `[{"start": float, "end": float, "speaker": str}, ...]`

    Raises:
        FileNotFoundError: If audio_path does not exist.
        RuntimeError: If the pipeline cannot be loaded.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    token = hf_token or os.getenv("HF_TOKEN")
    resolved_device = _resolve_device(device)
    pipeline = _load_pipeline(model, token, resolved_device)

    constraints: dict[str, int] = {}
    if num_speakers is not None:
        constraints["num_speakers"] = num_speakers
        logger.info("Running diarization on %s (num_speakers=%d)", audio_path.name, num_speakers)
    else:
        constraints["min_speakers"] = min_speakers
        constraints["max_speakers"] = max_speakers
        logger.info(
            "Running diarization on %s (min_speakers=%d, max_speakers=%d)",
            audio_path.name,
            min_speakers,
            max_speakers,
        )

    started = time.monotonic()
    with _progress_hook() as hook:
        if hook is not None:
            constraints["hook"] = hook
        diarization = pipeline(str(audio_path), **constraints)

    segments = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda segment: segment["start"])

    speakers = sorted({segment["speaker"] for segment in segments})
    logger.info(
        "Diarization complete in %.1f s: %d segments, %d speaker(s): %s",
        time.monotonic() - started,
        len(segments),
        len(speakers),
        ", ".join(speakers) or "none",
    )
    return segments


class _NullHook:
    """Stand-in used when pyannote's progress hook is unavailable."""

    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


def _progress_hook():
    """Returns pyannote's ProgressHook if installed, else a no-op context.

    Diarizing two hours of audio on CPU takes long enough that a progress
    bar is the difference between "working" and "apparently frozen".
    """
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        return ProgressHook()
    except Exception:       # pragma: no cover - depends on the install
        return _NullHook()
