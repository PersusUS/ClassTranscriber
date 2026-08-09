"""Shared audio helpers.

Small, dependency-light numeric utilities used by the recorder and the
preprocessor. Everything here works on plain numpy arrays and streams
files block by block, so a two-hour lecture never has to fit in RAM.
"""

import heapq
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_EPS = 1e-12


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapses a multi-channel array to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def rms(audio: np.ndarray) -> float:
    """Returns the root-mean-square amplitude of a signal."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def dbfs(audio: np.ndarray) -> float:
    """Returns the RMS level of a signal in dB relative to full scale."""
    return 20.0 * np.log10(max(rms(audio), _EPS))


def db_to_gain(db: float) -> float:
    """Converts a decibel value to a linear gain factor."""
    return float(10.0 ** (db / 20.0))


def peak_dbfs(audio: np.ndarray) -> float:
    """Returns the peak level of a signal in dBFS."""
    if audio.size == 0:
        return -np.inf
    return 20.0 * np.log10(max(float(np.max(np.abs(audio))), _EPS))


def design_highpass(sample_rate: int, cutoff_hz: float, order: int = 4):
    """Designs a Butterworth high-pass filter as second-order sections.

    Args:
        sample_rate: Sample rate of the audio in Hz.
        cutoff_hz: -3 dB cutoff frequency in Hz.
        order: Filter order. 4 attenuates 50 Hz mains hum by ~16 dB below
            an 80 Hz cutoff — a 2nd-order filter only manages ~9 dB, which
            leaves plenty of hum in the level measurement — while staying
            flat across the whole speech band.

    Returns:
        The filter in `sos` form, ready for `scipy.signal.sosfilt`.
    """
    from scipy.signal import butter

    nyquist = sample_rate / 2.0
    normalised = min(max(cutoff_hz / nyquist, 1e-4), 0.99)
    return butter(order, normalised, btype="highpass", output="sos")


def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Applies a zero-state high-pass filter to a whole array."""
    from scipy.signal import sosfilt

    sos = design_highpass(sample_rate, cutoff_hz)
    return sosfilt(sos, audio).astype(np.float32)


def soft_limit(audio: np.ndarray, ceiling: float = 0.97) -> np.ndarray:
    """Tames peaks above `ceiling` with a tanh knee instead of clipping.

    Hard clipping creates broadband harmonics that Whisper reads as
    consonants; a soft knee keeps loud syllables intelligible.
    """
    if audio.size == 0:
        return audio
    if float(np.max(np.abs(audio))) <= ceiling:
        return audio

    # Compress only the range above the ceiling. The knee is defined purely
    # in terms of `ceiling`, never the block's own peak, so the same input
    # sample always maps to the same output regardless of block boundaries.
    headroom = max(1.0 - ceiling, _EPS)
    over = np.abs(audio) > ceiling
    out = audio.astype(np.float32, copy=True)
    magnitude = np.abs(audio[over])
    out[over] = np.sign(audio[over]) * (
        ceiling + headroom * np.tanh((magnitude - ceiling) / headroom)
    )
    return out


def iter_blocks(
    path: Path,
    block_frames: int,
    overlap_frames: int = 0,
) -> Iterator[tuple[np.ndarray, int]]:
    """Streams a sound file as mono blocks with optional overlap.

    Args:
        path: Path to the audio file.
        block_frames: Number of frames per yielded block, excluding overlap.
        overlap_frames: Extra frames carried over from the previous block,
            prepended to each block after the first.

    Yields:
        Tuples of `(block, offset)` where `offset` is the frame index of
        the first *new* sample in the block.
    """
    with sf.SoundFile(str(path)) as handle:
        tail = np.zeros(0, dtype=np.float32)
        offset = 0
        while True:
            chunk = handle.read(block_frames, dtype="float32", always_2d=False)
            if chunk is None or len(chunk) == 0:
                break
            chunk = to_mono(np.asarray(chunk, dtype=np.float32))
            block = np.concatenate([tail, chunk]) if tail.size else chunk
            yield block, offset
            offset += len(chunk)
            if overlap_frames > 0:
                tail = chunk[-overlap_frames:].copy()


def crossfade(previous_tail: np.ndarray, next_head: np.ndarray) -> np.ndarray:
    """Linearly crossfades two equal-length overlapping regions.

    Used to hide the seams between independently denoised blocks.
    """
    length = min(len(previous_tail), len(next_head))
    if length == 0:
        return next_head
    ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
    blended = previous_tail[-length:] * (1.0 - ramp) + next_head[:length] * ramp
    return np.concatenate([blended, next_head[length:]]).astype(np.float32)


def scan_levels(
    path: Path,
    block_seconds: float = 1.0,
    noise_seconds: float = 2.0,
    highpass_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """First analysis pass: measures loudness and samples the room noise.

    Walks the whole file once, recording the RMS of every short block and
    keeping the quietest blocks as a fingerprint of the background noise.
    In a lecture hall the quietest moments are pauses, which is exactly
    the noise we want spectral gating to subtract.

    Args:
        path: Audio file to analyse.
        block_seconds: Analysis resolution.
        noise_seconds: How much quiet audio to collect as the profile.
        highpass_hz: When set, measure the audio as it will sound *after*
            high-pass filtering. Mains hum and air-conditioning rumble
            carry a lot of energy but no speech, so measuring before the
            filter overestimates the speech level and under-amplifies the
            professor.

    Returns:
        A tuple `(block_rms, noise_profile, sample_rate)`.
    """
    info = sf.info(str(path))
    sample_rate = info.samplerate
    block_frames = max(1, int(block_seconds * sample_rate))
    needed = max(1, int(np.ceil(noise_seconds / max(block_seconds, _EPS))))

    sos = design_highpass(sample_rate, highpass_hz) if highpass_hz > 0 else None
    filter_state = None
    if sos is not None:
        from scipy.signal import sosfilt, sosfilt_zi

    levels: list[float] = []
    # Max-heap (via negated keys) holding only the quietest `needed` blocks.
    # Keeping every block instead would put the entire recording in RAM,
    # which is precisely what the streaming design exists to avoid.
    quietest: list[tuple[float, int, np.ndarray]] = []

    for index, (block, _) in enumerate(iter_blocks(path, block_frames)):
        if sos is not None:
            if filter_state is None:
                filter_state = sosfilt_zi(sos) * block[0]
            block, filter_state = sosfilt(sos, block, zi=filter_state)
            block = block.astype(np.float32)

        level = rms(block)
        levels.append(level)

        if len(quietest) < needed:
            heapq.heappush(quietest, (-level, index, block))
        elif -level > quietest[0][0]:
            heapq.heapreplace(quietest, (-level, index, block))

    block_rms = np.asarray(levels, dtype=np.float64)
    if block_rms.size == 0:
        return block_rms, np.zeros(0, dtype=np.float32), sample_rate

    # Restore chronological order so the profile is contiguous audio.
    noise_profile = np.concatenate(
        [block for _, _, block in sorted(quietest, key=lambda item: item[1])]
    )

    return block_rms, noise_profile.astype(np.float32), sample_rate


def compute_gain(
    block_rms: np.ndarray,
    target_dbfs: float = -20.0,
    max_gain_db: float = 25.0,
) -> float:
    """Derives a single normalisation gain from a loudness scan.

    Uses the 75th percentile of block loudness as a robust estimate of
    *speech* level. The mean would be dragged down by long pauses and the
    peak would be set by a slammed door, so both would misjudge how much
    the professor's voice needs lifting.

    Args:
        block_rms: Per-block RMS values from `scan_levels`.
        target_dbfs: Desired speech level in dBFS.
        max_gain_db: Upper bound on amplification.

    Returns:
        A linear gain factor, never above `max_gain_db`.
    """
    if block_rms.size == 0:
        return 1.0

    speech_level = float(np.percentile(block_rms, 75))
    if speech_level <= _EPS:
        return 1.0

    speech_dbfs = 20.0 * np.log10(speech_level)
    gain_db = min(target_dbfs - speech_dbfs, max_gain_db)
    return db_to_gain(gain_db)


def format_timestamp(seconds: float, with_millis: bool = False) -> str:
    """Formats seconds as HH:MM:SS (optionally HH:MM:SS,mmm for SRT)."""
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if with_millis:
        millis = int(round((seconds - total) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
