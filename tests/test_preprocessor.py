"""Unit tests for M2 — modules/preprocessor.py."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pytest import approx

from modules.audio_utils import dbfs, rms
from modules.preprocessor import ensure_mono_16k, preprocess


def _write(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> Path:
    sf.write(str(path), audio, sample_rate, subtype="FLOAT")
    return path


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """One second of noise at 16 kHz mono."""
    rng = np.random.default_rng(42)
    return _write(tmp_path / "input.wav", rng.uniform(-0.5, 0.5, 16000).astype(np.float32))


@pytest.fixture
def speech_like(tmp_path: Path) -> Path:
    """Ten seconds of quiet 'speech' bursts over a constant noise floor."""
    sample_rate = 16000
    rng = np.random.default_rng(7)
    audio = rng.normal(0, 0.002, sample_rate * 10).astype(np.float32)
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    for start in (1, 3, 5, 7):
        burst = 0.02 * np.sin(2 * np.pi * 220 * time).astype(np.float32)
        audio[start * sample_rate : (start + 1) * sample_rate] += burst
    return _write(tmp_path / "speech.wav", audio)


def test_preprocess_creates_output_file(sample_wav: Path, tmp_path: Path):
    """Assert the output file is created."""
    output = tmp_path / "output.wav"
    preprocess(sample_wav, output)
    assert output.exists()


def test_preprocess_invalid_input(tmp_path: Path):
    """Assert FileNotFoundError for a missing input."""
    with pytest.raises(FileNotFoundError):
        preprocess(tmp_path / "nonexistent.wav", tmp_path / "output.wav")


def test_preprocess_output_normalized(sample_wav: Path, tmp_path: Path):
    """The output never exceeds full scale."""
    output = tmp_path / "output.wav"
    preprocess(sample_wav, output)
    audio, _ = sf.read(str(output), dtype="float32")
    assert np.max(np.abs(audio)) <= 1.0


def test_preprocess_same_samplerate(sample_wav: Path, tmp_path: Path):
    """16 kHz input stays at 16 kHz."""
    output = tmp_path / "output.wav"
    preprocess(sample_wav, output)
    assert sf.info(str(output)).samplerate == sf.info(str(sample_wav)).samplerate


def test_preprocess_preserves_duration(speech_like: Path, tmp_path: Path):
    """Block processing must not add or drop samples — timestamps depend on it."""
    output = tmp_path / "output.wav"
    preprocess(speech_like, output, denoise=False)

    assert sf.info(str(output)).frames == sf.info(str(speech_like)).frames


def test_preprocess_lifts_quiet_speech(speech_like: Path, tmp_path: Path):
    """A distant professor gets amplified towards the target level.

    Peak normalisation, which the original used, would barely move this
    file because the noise already touches a high instantaneous peak.
    """
    output = tmp_path / "output.wav"
    preprocess(speech_like, output, denoise=False)

    audio, _ = sf.read(str(output), dtype="float32")

    assert dbfs(audio) > dbfs(sf.read(str(speech_like), dtype="float32")[0]) + 10


def test_preprocess_removes_low_frequency_rumble(tmp_path: Path):
    """A 30 Hz air-conditioning rumble is filtered out; the voice band is not."""
    sample_rate = 16000
    time = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    rumble = 0.4 * np.sin(2 * np.pi * 30 * time).astype(np.float32)
    voice = 0.1 * np.sin(2 * np.pi * 300 * time).astype(np.float32)
    source = _write(tmp_path / "rumble.wav", rumble + voice)

    output = tmp_path / "clean.wav"
    preprocess(source, output, denoise=False)
    audio, _ = sf.read(str(output), dtype="float32")

    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    rumble_energy = spectrum[np.argmin(np.abs(frequencies - 30))]
    voice_energy = spectrum[np.argmin(np.abs(frequencies - 300))]

    assert rumble_energy < voice_energy


def test_preprocess_rejects_empty_audio(tmp_path: Path):
    """An empty recording is reported instead of producing an empty transcript."""
    source = _write(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32))

    with pytest.raises(ValueError, match="no audio"):
        preprocess(source, tmp_path / "output.wav")


def test_ensure_mono_16k_passes_through(sample_wav: Path, tmp_path: Path):
    """Audio already in the target format is not rewritten."""
    assert ensure_mono_16k(sample_wav, tmp_path / "converted.wav") == sample_wav


def test_ensure_mono_16k_resamples_and_downmixes(tmp_path: Path):
    """A 44.1 kHz stereo phone recording is converted, keeping its duration."""
    sample_rate = 44100
    rng = np.random.default_rng(3)
    stereo = rng.uniform(-0.3, 0.3, (sample_rate * 2, 2)).astype(np.float32)
    source = _write(tmp_path / "phone.wav", stereo, sample_rate)

    converted = ensure_mono_16k(source, tmp_path / "converted.wav")
    info = sf.info(str(converted))

    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.frames / info.samplerate == approx(2.0, rel=0.01)


def test_preprocess_accepts_stereo_input(tmp_path: Path):
    """Stereo input is downmixed rather than rejected."""
    sample_rate = 44100
    rng = np.random.default_rng(11)
    stereo = rng.uniform(-0.2, 0.2, (sample_rate, 2)).astype(np.float32)
    source = _write(tmp_path / "stereo.wav", stereo, sample_rate)

    output = tmp_path / "clean.wav"
    preprocess(source, output, denoise=False)

    assert sf.info(str(output)).channels == 1
    assert sf.info(str(output)).samplerate == 16000


def test_preprocess_denoise_reduces_steady_noise(tmp_path: Path):
    """Spectral gating attenuates a steady hiss when noisereduce is installed."""
    pytest.importorskip("noisereduce")

    sample_rate = 16000
    rng = np.random.default_rng(5)
    time = np.arange(sample_rate * 6, dtype=np.float32) / sample_rate
    noise = rng.normal(0, 0.05, time.size).astype(np.float32)
    voice = np.zeros_like(noise)
    voice[sample_rate * 2 : sample_rate * 4] = 0.2 * np.sin(
        2 * np.pi * 300 * time[: sample_rate * 2]
    )
    source = _write(tmp_path / "noisy.wav", noise + voice)

    denoised = tmp_path / "denoised.wav"
    plain = tmp_path / "plain.wav"
    preprocess(source, denoised, denoise=True)
    preprocess(source, plain, denoise=False)

    silent_slice = slice(0, sample_rate)
    denoised_audio, _ = sf.read(str(denoised), dtype="float32")
    plain_audio, _ = sf.read(str(plain), dtype="float32")

    assert rms(denoised_audio[silent_slice]) < rms(plain_audio[silent_slice])
