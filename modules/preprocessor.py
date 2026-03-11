"""M2 — Noise reduction and volume normalization module.

Applies spectral gating noise reduction and amplitude normalization
to a WAV file to improve quality before diarization and transcription.
"""

import logging
from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf

logger = logging.getLogger(__name__)


def preprocess(
    input_path: Path,
    output_path: Path,
    stationary: bool = False,
) -> Path:
    """Loads a WAV file, applies spectral gating noise reduction,
    normalizes the volume, and saves the result to output_path.

    Args:
        input_path: Path to the source WAV file.
        output_path: Destination path for the processed WAV file.
        stationary: If False (default), uses non-stationary mode
            which is recommended for classroom audio where background
            noise changes over time.

    Returns:
        The path to the saved output file.

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    logger.info("Loading audio from %s", input_path)
    audio, sr = sf.read(input_path, dtype="float32")

    logger.info(
        "Applying %s noise reduction (prop_decrease=0.75)",
        "stationary" if stationary else "non-stationary",
    )
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=stationary,
        prop_decrease=0.75,
    )

    logger.info("Normalizing amplitude")
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    sf.write(str(output_path), audio, sr)
    logger.info("Preprocessed audio saved to %s", output_path)

    return output_path
