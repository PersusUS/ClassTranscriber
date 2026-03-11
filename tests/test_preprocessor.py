"""Unit tests for M2 — modules/preprocessor.py."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from modules.preprocessor import preprocess


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Creates a short synthetic WAV file for testing."""
    path = tmp_path / "input.wav"
    # 1 second of random noise at 16kHz mono
    rng = np.random.default_rng(42)
    audio = rng.uniform(-0.5, 0.5, 16000).astype(np.float32)
    sf.write(str(path), audio, 16000)
    return path


def test_preprocess_creates_output_file(sample_wav: Path, tmp_path: Path):
    """Assert output file is created after calling preprocess()."""
    output_path = tmp_path / "output.wav"
    preprocess(sample_wav, output_path)
    assert output_path.exists()


def test_preprocess_invalid_input(tmp_path: Path):
    """Pass a nonexistent input path. Assert FileNotFoundError is raised."""
    bad_input = tmp_path / "nonexistent.wav"
    output_path = tmp_path / "output.wav"
    with pytest.raises(FileNotFoundError):
        preprocess(bad_input, output_path)


def test_preprocess_output_normalized(sample_wav: Path, tmp_path: Path):
    """Load the output file and assert np.max(np.abs(audio)) <= 1.0."""
    output_path = tmp_path / "output.wav"
    preprocess(sample_wav, output_path)
    audio, _ = sf.read(str(output_path), dtype="float32")
    assert np.max(np.abs(audio)) <= 1.0


def test_preprocess_same_samplerate(sample_wav: Path, tmp_path: Path):
    """Assert input and output have the same sample rate."""
    output_path = tmp_path / "output.wav"
    preprocess(sample_wav, output_path)
    input_info = sf.info(str(sample_wav))
    output_info = sf.info(str(output_path))
    assert input_info.samplerate == output_info.samplerate
