"""Unit tests for modules/audio_utils.py."""

from pathlib import Path

import numpy as np
import soundfile as sf
from pytest import approx

from modules.audio_utils import (
    compute_gain,
    crossfade,
    dbfs,
    db_to_gain,
    format_timestamp,
    highpass,
    iter_blocks,
    rms,
    scan_levels,
    soft_limit,
    to_mono,
)


def test_to_mono_averages_channels():
    """Stereo collapses to the average of both channels."""
    stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    assert np.allclose(to_mono(stereo), [0.5, 0.5])


def test_rms_and_dbfs_of_full_scale_sine():
    """A full-scale sine sits at about -3 dBFS RMS."""
    time = np.linspace(0, 1, 16000, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    assert rms(sine) == approx(1 / np.sqrt(2), 0.01)
    assert dbfs(sine) == approx(-3.01, 0.1)


def test_dbfs_of_silence_is_finite():
    """Silence must not produce -inf or a division warning."""
    assert dbfs(np.zeros(100, dtype=np.float32)) < -200


def test_db_to_gain_roundtrip():
    """+6 dB roughly doubles the amplitude."""
    assert db_to_gain(6.0) == approx(2.0, 0.01)


def test_highpass_removes_low_frequency_rumble():
    """An 80 Hz filter kills a 20 Hz rumble and keeps a 1 kHz tone."""
    sample_rate = 16000
    time = np.linspace(0, 1, sample_rate, endpoint=False)
    rumble = np.sin(2 * np.pi * 20 * time).astype(np.float32)
    tone = np.sin(2 * np.pi * 1000 * time).astype(np.float32)

    filtered_rumble = highpass(rumble, sample_rate, 80.0)
    filtered_tone = highpass(tone, sample_rate, 80.0)

    assert rms(filtered_rumble) < 0.15 * rms(rumble)
    assert rms(filtered_tone) > 0.9 * rms(tone)


def test_soft_limit_stays_below_one():
    """Peaks above the ceiling are compressed, never clipped past 1.0."""
    loud = np.array([-3.0, -1.2, 0.0, 0.5, 1.5, 4.0], dtype=np.float32)

    limited = soft_limit(loud, ceiling=0.97)

    assert np.max(np.abs(limited)) <= 1.0       # tanh asymptotes at full scale
    assert abs(limited[1]) < abs(loud[1])       # 1.2 is pulled back under 1.0
    assert limited[3] == approx(0.5, 1e-6)      # Below the ceiling: untouched
    assert limited[0] < 0                       # Sign is preserved


def test_soft_limit_is_block_independent():
    """The same sample maps to the same output regardless of its neighbours.

    Blocks are processed separately, so a limiter that scaled by the local
    peak would create audible steps at the seams.
    """
    quiet_block = np.array([0.99, 0.5], dtype=np.float32)
    loud_block = np.array([0.99, 4.0], dtype=np.float32)

    assert soft_limit(quiet_block)[0] == approx(soft_limit(loud_block)[0], 1e-6)


def test_crossfade_is_continuous():
    """The blend starts at the old signal and ends at the new one."""
    previous = np.ones(100, dtype=np.float32)
    following = np.zeros(100, dtype=np.float32)

    blended = crossfade(previous, following)

    assert blended[0] == approx(1.0, 1e-6)
    assert blended[-1] == approx(0.0, 1e-6)
    assert len(blended) == 100


def test_iter_blocks_covers_every_sample(tmp_path: Path):
    """Streaming in blocks yields each sample exactly once."""
    audio = np.arange(1000, dtype=np.float32) / 1000.0
    path = tmp_path / "ramp.wav"
    sf.write(str(path), audio, 16000, subtype="FLOAT")

    collected = []
    for block, offset in iter_blocks(path, block_frames=256):
        collected.append(block)

    assert len(np.concatenate(collected)) == 1000


def test_iter_blocks_overlap_offsets(tmp_path: Path):
    """With overlap, offsets still advance by the number of new samples."""
    audio = np.zeros(1000, dtype=np.float32)
    path = tmp_path / "silence.wav"
    sf.write(str(path), audio, 16000, subtype="FLOAT")

    offsets = [offset for _, offset in iter_blocks(path, block_frames=400, overlap_frames=50)]

    assert offsets == [0, 400, 800]


def test_scan_levels_finds_the_quiet_part(tmp_path: Path):
    """The noise profile is sampled from the quietest audio, not the loudest."""
    sample_rate = 16000
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 0.01, sample_rate * 2).astype(np.float32)
    loud = rng.normal(0, 0.3, sample_rate * 2).astype(np.float32)
    path = tmp_path / "mixed.wav"
    sf.write(str(path), np.concatenate([loud, quiet]), sample_rate, subtype="FLOAT")

    block_rms, noise_profile, rate = scan_levels(path, block_seconds=1.0, noise_seconds=2.0)

    assert rate == sample_rate
    assert len(block_rms) == 4
    assert rms(noise_profile) < 0.05


def test_scan_levels_profile_is_bounded(tmp_path: Path):
    """The noise profile stays small no matter how long the recording is.

    A two-hour lecture must not end up held in RAM just to pick its
    quietest moments.
    """
    sample_rate = 16000
    rng = np.random.default_rng(2)
    audio = rng.normal(0, 0.05, sample_rate * 60).astype(np.float32)
    path = tmp_path / "long.wav"
    sf.write(str(path), audio, sample_rate, subtype="FLOAT")

    block_rms, noise_profile, _ = scan_levels(path, block_seconds=1.0, noise_seconds=2.0)

    assert len(block_rms) == 60
    assert len(noise_profile) == 2 * sample_rate


def test_scan_levels_with_highpass_ignores_hum(tmp_path: Path):
    """Measuring after the high-pass keeps mains hum out of the level estimate.

    Otherwise a loud 50 Hz hum reads as a loud room and the professor
    never gets amplified.
    """
    sample_rate = 16000
    time = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    hum = 0.5 * np.sin(2 * np.pi * 50 * time).astype(np.float32)
    path = tmp_path / "hum.wav"
    sf.write(str(path), hum, sample_rate, subtype="FLOAT")

    raw_levels, _, _ = scan_levels(path, block_seconds=1.0)
    filtered_levels, _, _ = scan_levels(path, block_seconds=1.0, highpass_hz=80.0)

    assert float(np.median(filtered_levels)) < 0.2 * float(np.median(raw_levels))


def test_compute_gain_lifts_a_quiet_lecture():
    """Quiet speech gets amplified towards the target level."""
    quiet = np.full(20, 0.01)      # -40 dBFS

    gain = compute_gain(quiet, target_dbfs=-20.0, max_gain_db=25.0)

    assert gain == approx(10.0, 0.01)


def test_compute_gain_respects_the_ceiling():
    """Amplification is capped so noise is not blown up with the voice."""
    very_quiet = np.full(20, 0.0001)

    assert compute_gain(very_quiet, max_gain_db=25.0) == approx(db_to_gain(25.0), 0.01)


def test_compute_gain_ignores_silence_between_sentences():
    """Long pauses must not drag the estimate down and cause over-amplification.

    Peak- or mean-based normalisation both get this wrong; the 75th
    percentile tracks the speech, not the gaps.
    """
    mostly_silence = np.array([0.0001] * 60 + [0.05] * 40)

    gain = compute_gain(mostly_silence, target_dbfs=-20.0)

    assert gain == approx(2.0, 0.05)


def test_format_timestamp():
    """Timestamps render as HH:MM:SS, and as HH:MM:SS,mmm for SRT."""
    assert format_timestamp(3725.0) == "01:02:05"
    assert format_timestamp(3725.5, with_millis=True) == "01:02:05,500"
    assert format_timestamp(-5.0) == "00:00:00"
