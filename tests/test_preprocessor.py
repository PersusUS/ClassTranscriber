"""Unit tests for M2 — modules/preprocessor.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pytest import approx

import config
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


def test_preprocess_without_noisereduce(tmp_path: Path, monkeypatch, caplog):
    """Missing noisereduce degrades to filter + normalise instead of failing.

    The high-pass and the loudness normalisation are the two stages that
    matter most, so losing spectral gating must not lose the whole run.
    """
    import modules.preprocessor as preprocessor_mod

    monkeypatch.setitem(sys.modules, "noisereduce", None)
    monkeypatch.setattr(preprocessor_mod._reduce_noise, "_warned", False, raising=False)

    rng = np.random.default_rng(13)
    source = _write(tmp_path / "in.wav", rng.uniform(-0.3, 0.3, 32000).astype(np.float32))
    output = tmp_path / "out.wav"

    with caplog.at_level("WARNING"):
        preprocess(source, output, denoise=True)

    assert output.exists()
    assert sf.info(str(output)).frames == 32000
    assert "noisereduce is not installed" in caplog.text


def test_preprocess_warns_on_low_snr(tmp_path: Path, caplog):
    """A voice buried in noise is flagged so the microphone can be moved."""
    sample_rate = 16000
    rng = np.random.default_rng(21)
    # Loud, constant noise with a barely-there voice on top: SNR near zero.
    audio = rng.normal(0, 0.1, sample_rate * 10).astype(np.float32)
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio[: sample_rate] += 0.01 * np.sin(2 * np.pi * 300 * time).astype(np.float32)
    source = _write(tmp_path / "buried.wav", audio)

    with caplog.at_level("WARNING"):
        preprocess(source, tmp_path / "out.wav", denoise=False)

    assert "signal-to-noise" in caplog.text


def test_preprocess_quiet_on_good_snr(speech_like: Path, tmp_path: Path, caplog):
    """Clear speech over a low noise floor produces no SNR warning."""
    with caplog.at_level("WARNING"):
        preprocess(speech_like, tmp_path / "out.wav", denoise=False)

    assert "signal-to-noise" not in caplog.text


def test_preprocess_can_disable_highpass(tmp_path: Path):
    """highpass_hz=0 leaves the low end untouched."""
    sample_rate = 16000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    rumble = 0.3 * np.sin(2 * np.pi * 30 * time).astype(np.float32)
    source = _write(tmp_path / "rumble.wav", rumble)

    output = tmp_path / "out.wav"
    preprocess(source, output, denoise=False, highpass_hz=0)

    audio, _ = sf.read(str(output), dtype="float32")
    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(len(audio), 1 / sample_rate)

    assert spectrum[np.argmin(np.abs(frequencies - 30))] > 0.1 * spectrum.max()


@pytest.fixture
def small_blocks(monkeypatch):
    """Shrinks the block size so multi-block processing is exercised fast.

    With the real 30-second blocks, only recordings longer than that reach
    the overlap-and-crossfade path — which is the path every actual
    two-hour lecture takes.
    """
    monkeypatch.setattr(config, "PREPROCESS_BLOCK_SECONDS", 1.0)
    monkeypatch.setattr(config, "PREPROCESS_OVERLAP_SECONDS", 0.25)


def _tone(seconds: float, sample_rate: int = 16000, freq: float = 300.0) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.2 * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def test_preprocess_multi_block_preserves_frames(tmp_path: Path, small_blocks):
    """Crossing many block seams must not gain or lose a single sample."""
    source = _write(tmp_path / "long.wav", _tone(5.0))
    output = tmp_path / "out.wav"

    preprocess(source, output, denoise=False)

    assert sf.info(str(output)).frames == sf.info(str(source)).frames


def test_preprocess_final_partial_block(tmp_path: Path, small_blocks):
    """A recording that does not end on a block boundary is written in full."""
    source = _write(tmp_path / "odd.wav", _tone(5.1))
    output = tmp_path / "out.wav"

    preprocess(source, output, denoise=False)

    assert sf.info(str(output)).frames == sf.info(str(source)).frames


def test_preprocess_block_size_does_not_change_the_result(tmp_path: Path, monkeypatch):
    """Block boundaries must be invisible in the output.

    The same audio processed in one block and in five must come out the
    same, or the seams are colouring the signal.
    """
    source = _write(tmp_path / "tone.wav", _tone(5.0))

    monkeypatch.setattr(config, "PREPROCESS_BLOCK_SECONDS", 30.0)
    single = tmp_path / "single.wav"
    preprocess(source, single, denoise=False)

    monkeypatch.setattr(config, "PREPROCESS_BLOCK_SECONDS", 1.0)
    monkeypatch.setattr(config, "PREPROCESS_OVERLAP_SECONDS", 0.25)
    chunked = tmp_path / "chunked.wav"
    preprocess(source, chunked, denoise=False)

    single_audio, _ = sf.read(str(single), dtype="float32")
    chunked_audio, _ = sf.read(str(chunked), dtype="float32")

    assert len(single_audio) == len(chunked_audio)
    assert np.max(np.abs(single_audio - chunked_audio)) < 1e-3


def test_preprocess_multi_block_has_no_seam_clicks(tmp_path: Path, small_blocks):
    """Denoised blocks are crossfaded, so no click appears at the seams.

    Independently gated blocks stitched end to end would leave a step
    discontinuity every block, audible as a tick and harmful to the VAD.
    """
    pytest.importorskip("noisereduce")

    sample_rate = 16000
    rng = np.random.default_rng(31)
    audio = _tone(5.0) + rng.normal(0, 0.01, sample_rate * 5).astype(np.float32)
    source = _write(tmp_path / "noisy.wav", audio)
    output = tmp_path / "out.wav"

    preprocess(source, output, denoise=True)
    processed, _ = sf.read(str(output), dtype="float32")

    jumps = np.abs(np.diff(processed))
    seam_indices = [
        index for index in range(sample_rate, len(processed) - 1, sample_rate)
    ]
    seam_jumps = jumps[seam_indices]

    # A seam must not stand out against the signal's own sample-to-sample motion.
    assert float(np.max(seam_jumps)) < 5.0 * float(np.percentile(jumps, 99))
