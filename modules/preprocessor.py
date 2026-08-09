"""M2 — Noise reduction and loudness normalization.

Built for the actual problem: a laptop microphone in a room full of air
conditioning, coughing, chair scraping and side conversations, recording
a professor who is several metres away.

The chain is, in order:

1. **Resample to mono 16 kHz** if needed, so phone recordings work too.
2. **High-pass filter at 80 Hz** — removes HVAC rumble, projector hum and
   desk thumps. None of that overlaps speech, and it is what makes the
   later gain stage safe to apply.
3. **Spectral gating** against a noise fingerprint measured from the
   quietest moments of *this* recording, rather than a generic assumption.
4. **RMS normalization** to a target loudness, with a soft limiter.

Everything streams block by block, so memory stays flat regardless of
whether the recording is five minutes or three hours.
"""

import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import config
from modules.audio_utils import (
    compute_gain,
    crossfade,
    design_highpass,
    scan_levels,
    soft_limit,
    to_mono,
)

logger = logging.getLogger(__name__)


def _reduce_noise(
    block: np.ndarray,
    sample_rate: int,
    noise_profile: np.ndarray | None,
    prop_decrease: float,
    stationary: bool,
) -> np.ndarray:
    """Applies spectral gating to one block, or returns it untouched.

    `noisereduce` is optional: if it is not installed the rest of the
    chain (high-pass, normalisation) is still worth running, so this
    warns once instead of failing the whole pipeline.
    """
    try:
        import noisereduce as nr
    except ImportError:
        if not getattr(_reduce_noise, "_warned", False):
            logger.warning(
                "noisereduce is not installed — skipping spectral gating. "
                "Install it with: pip install noisereduce"
            )
            _reduce_noise._warned = True
        return block

    kwargs = {
        "y": block,
        "sr": sample_rate,
        "stationary": stationary,
        "prop_decrease": prop_decrease,
    }
    if noise_profile is not None and noise_profile.size >= sample_rate // 4:
        kwargs["y_noise"] = noise_profile

    return np.asarray(nr.reduce_noise(**kwargs), dtype=np.float32)


def ensure_mono_16k(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int = config.SAMPLE_RATE,
) -> Path:
    """Converts a recording to mono at `target_sample_rate` if necessary.

    Streams the file in blocks and resamples with a polyphase filter.
    Block sizes are chosen as exact multiples of the decimation factor so
    no frames are gained or lost, which keeps timestamps aligned with the
    original recording.

    Args:
        input_path: Source audio file (any format soundfile can read).
        output_path: Destination WAV path.
        target_sample_rate: Desired sample rate in Hz.

    Returns:
        `input_path` unchanged if it is already mono at the target rate,
        otherwise `output_path`.
    """
    info = sf.info(str(input_path))
    if info.samplerate == target_sample_rate and info.channels == 1:
        return input_path

    from scipy.signal import resample_poly

    divisor = math.gcd(info.samplerate, target_sample_rate)
    up = target_sample_rate // divisor
    down = info.samplerate // divisor

    logger.info(
        "Converting %s from %d Hz / %d ch to %d Hz mono",
        input_path.name,
        info.samplerate,
        info.channels,
        target_sample_rate,
    )

    # Keep the block an exact multiple of `down` so up/down stays integral.
    block_frames = max(down, (int(30.0 * info.samplerate) // down) * down)

    with sf.SoundFile(str(input_path)) as source, sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=target_sample_rate,
        channels=1,
        subtype=config.RECORD_SUBTYPE,
    ) as sink:
        while True:
            chunk = source.read(block_frames, dtype="float32", always_2d=False)
            if chunk is None or len(chunk) == 0:
                break
            mono = to_mono(np.asarray(chunk, dtype=np.float32))
            if up != down:
                mono = resample_poly(mono, up, down).astype(np.float32)
            sink.write(mono)

    return output_path


def preprocess(
    input_path: Path,
    output_path: Path,
    stationary: bool = True,
    denoise: bool = True,
    prop_decrease: float = config.NOISE_PROP_DECREASE,
    highpass_hz: float = config.HIGHPASS_HZ,
    target_dbfs: float = config.TARGET_DBFS,
) -> Path:
    """Cleans up a recording for diarization and transcription.

    Args:
        input_path: Path to the source audio file.
        output_path: Destination path for the processed WAV file.
        stationary: True (default) gates against a fixed noise fingerprint
            taken from the recording's quietest moments — cheaper and more
            stable than re-estimating noise continuously. Set False for
            recordings whose background changes drastically.
        denoise: Set False to skip spectral gating entirely (useful when
            the room is already quiet, or to save CPU).
        prop_decrease: Strength of the noise attenuation, 0.0–1.0.
        highpass_hz: High-pass cutoff in Hz. 0 disables the filter.
        target_dbfs: Target speech loudness in dBFS.

    Returns:
        The path to the saved output file.

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        source = ensure_mono_16k(input_path, workdir / "mono16k.wav")

        # Pass 1: measure the room. The high-pass is applied to the
        # measurement too, so mains hum does not masquerade as speech.
        logger.info("Analysing levels and room noise in %s", source.name)
        block_rms, noise_profile, sample_rate = scan_levels(
            source,
            noise_seconds=config.NOISE_PROFILE_SECONDS,
            highpass_hz=highpass_hz,
        )

        if block_rms.size == 0:
            raise ValueError(f"Input file contains no audio: {input_path}")

        noise_floor_db = 20.0 * np.log10(max(float(np.percentile(block_rms, 10)), 1e-12))
        speech_db = 20.0 * np.log10(max(float(np.percentile(block_rms, 75)), 1e-12))
        snr = speech_db - noise_floor_db
        logger.info(
            "Speech level %.1f dBFS, noise floor %.1f dBFS (SNR ~%.1f dB)",
            speech_db,
            noise_floor_db,
            snr,
        )
        if snr < 10.0:
            logger.warning(
                "Low signal-to-noise ratio (~%.1f dB). Diarization accuracy will suffer; "
                "consider a microphone closer to the professor.",
                snr,
            )

        # Pass 2: filter and denoise at unit gain.
        filtered = workdir / "filtered.wav"
        _process_stream(
            source=source,
            output_path=filtered,
            sample_rate=sample_rate,
            noise_profile=noise_profile if denoise else None,
            denoise=denoise,
            prop_decrease=prop_decrease,
            highpass_hz=highpass_hz,
            stationary=stationary,
        )

        # Pass 3: normalise against the *cleaned* signal. Denoising lowers
        # the overall level, so a gain derived from the raw audio would
        # leave the final file quieter than the target.
        cleaned_rms, _, _ = scan_levels(filtered)
        gain = compute_gain(cleaned_rms, target_dbfs=target_dbfs, max_gain_db=config.MAX_GAIN_DB)
        logger.info(
            "Applying %.1f dB of gain to reach %.0f dBFS",
            20.0 * np.log10(max(gain, 1e-12)),
            target_dbfs,
        )

        _apply_gain(filtered, output_path, sample_rate, gain)

    logger.info("Preprocessed audio saved to %s", output_path)
    return output_path


def _apply_gain(source: Path, output_path: Path, sample_rate: int, gain: float) -> None:
    """Streams a file through a fixed gain and the soft limiter."""
    block_frames = max(1, int(config.PREPROCESS_BLOCK_SECONDS * sample_rate))

    with sf.SoundFile(str(source)) as handle, sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=sample_rate,
        channels=1,
        subtype=config.RECORD_SUBTYPE,
    ) as sink:
        while True:
            chunk = handle.read(block_frames, dtype="float32", always_2d=False)
            if chunk is None or len(chunk) == 0:
                break
            sink.write(soft_limit(np.asarray(chunk, dtype=np.float32) * gain, config.PEAK_CEILING))


def _process_stream(
    source: Path,
    output_path: Path,
    sample_rate: int,
    noise_profile: np.ndarray | None,
    denoise: bool,
    prop_decrease: float,
    highpass_hz: float,
    stationary: bool,
) -> None:
    """Second pass: high-pass filters and denoises, writing at unit gain.

    Blocks are denoised with a shared overlap region and crossfaded, which
    hides the discontinuities that independent per-block spectral gating
    would otherwise leave at the seams.
    """
    from scipy.signal import sosfilt, sosfilt_zi

    block_frames = max(1, int(config.PREPROCESS_BLOCK_SECONDS * sample_rate))
    overlap_frames = max(1, int(config.PREPROCESS_OVERLAP_SECONDS * sample_rate))

    sos = design_highpass(sample_rate, highpass_hz) if highpass_hz > 0 else None
    filter_state = None

    previous_raw_tail = np.zeros(0, dtype=np.float32)
    pending = np.zeros(0, dtype=np.float32)
    written = 0

    # Float intermediate: filtering can push samples past full scale, and
    # clipping them here would defeat the limiter at the end of the chain.
    with sf.SoundFile(str(source)) as handle, sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=sample_rate,
        channels=1,
        subtype="FLOAT",
    ) as sink:

        def emit(samples: np.ndarray) -> None:
            nonlocal written
            if samples.size == 0:
                return
            sink.write(samples)
            written += samples.size

        while True:
            chunk = handle.read(block_frames, dtype="float32", always_2d=False)
            if chunk is None or len(chunk) == 0:
                break
            chunk = to_mono(np.asarray(chunk, dtype=np.float32))

            if sos is not None:
                # Carry the filter state across blocks so the high-pass is
                # continuous and leaves no transient at each seam.
                if filter_state is None:
                    filter_state = sosfilt_zi(sos) * chunk[0]
                chunk, filter_state = sosfilt(sos, chunk, zi=filter_state)
                chunk = chunk.astype(np.float32)

            if denoise:
                buffered = np.concatenate([previous_raw_tail, chunk])
                processed = _reduce_noise(
                    buffered, sample_rate, noise_profile, prop_decrease, stationary
                )
            else:
                processed = np.concatenate([previous_raw_tail, chunk])

            overlap_estimate = processed[: previous_raw_tail.size]
            fresh = processed[previous_raw_tail.size :]

            if pending.size:
                emit(crossfade(pending, overlap_estimate))
                pending = np.zeros(0, dtype=np.float32)

            if fresh.size > overlap_frames:
                emit(fresh[:-overlap_frames])
                pending = fresh[-overlap_frames:].copy()
            else:
                pending = fresh.copy()

            previous_raw_tail = chunk[-overlap_frames:].copy()

        emit(pending)

    logger.info("Wrote %.1f s of processed audio", written / sample_rate)
