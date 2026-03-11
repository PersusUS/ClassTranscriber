"""M1 — Audio capture module.

Records audio from the default microphone to disk in real time,
saving to a WAV file at 16kHz mono.
"""

import logging
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


def record(
    output_path: Path,
    duration_seconds: int,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Records audio from the default system microphone for the given duration.

    Saves to output_path as a 16kHz mono WAV file.
    Logs progress every 10 seconds.
    Returns the path to the saved file.

    Args:
        output_path: Destination path for the WAV file.
        duration_seconds: How long to record in seconds.
        sample_rate: Sample rate in Hz. Default 16000.
        channels: Number of audio channels. Default 1 (mono).

    Returns:
        The path to the saved WAV file.

    Raises:
        FileNotFoundError: If the output directory does not exist.
        RuntimeError: If no input device is found.
    """
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_path.parent}"
        )

    try:
        device_info = sd.query_devices(kind="input")
    except sd.PortAudioError as exc:
        raise RuntimeError("No input audio device found.") from exc

    logger.info("Recording device: %s", device_info["name"])
    logger.info(
        "Starting recording — %d seconds at %d Hz, %d channel(s)",
        duration_seconds,
        sample_rate,
        channels,
    )

    total_frames = int(duration_seconds * sample_rate)
    audio = sd.rec(
        total_frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
    )

    # Log progress every 10 seconds
    elapsed = 0
    while elapsed < duration_seconds:
        sleep_time = min(10, duration_seconds - elapsed)
        time.sleep(sleep_time)
        elapsed += sleep_time
        logger.info("Recording progress: %d / %d seconds", elapsed, duration_seconds)

    sd.wait()

    sf.write(str(output_path), audio, sample_rate)
    logger.info("Recording saved to %s", output_path)

    return output_path
