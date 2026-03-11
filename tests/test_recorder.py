"""Unit tests for M1 — modules/recorder.py."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import soundfile as sf

from modules.recorder import record


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Provides a temporary directory for test output files."""
    return tmp_path


@patch("modules.recorder.sd")
def test_record_creates_file(mock_sd, output_dir: Path):
    """Mock sounddevice.rec and sounddevice.wait. Assert output WAV file is created."""
    output_path = output_dir / "test.wav"

    # Make sd.rec return a numpy array of the right shape
    mock_sd.rec.return_value = np.zeros((16000 * 2, 1), dtype=np.float32)
    mock_sd.query_devices.return_value = {"name": "Test Microphone"}

    result = record(output_path, duration_seconds=2, sample_rate=16000, channels=1)

    assert output_path.exists()


@patch("modules.recorder.sd")
def test_record_invalid_directory(mock_sd):
    """Pass a path with a nonexistent parent directory. Assert FileNotFoundError."""
    bad_path = Path("/nonexistent/directory/test.wav")

    with pytest.raises(FileNotFoundError):
        record(bad_path, duration_seconds=2)


@patch("modules.recorder.sd")
def test_record_returns_path(mock_sd, output_dir: Path):
    """Assert the return value equals the given output_path."""
    output_path = output_dir / "test.wav"

    mock_sd.rec.return_value = np.zeros((16000 * 2, 1), dtype=np.float32)
    mock_sd.query_devices.return_value = {"name": "Test Microphone"}

    result = record(output_path, duration_seconds=2, sample_rate=16000, channels=1)

    assert result == output_path


@patch("modules.recorder.sd")
def test_record_correct_samplerate(mock_sd, output_dir: Path):
    """Use soundfile.info() on the output file to assert samplerate == 16000."""
    output_path = output_dir / "test.wav"

    mock_sd.rec.return_value = np.zeros((16000 * 2, 1), dtype=np.float32)
    mock_sd.query_devices.return_value = {"name": "Test Microphone"}

    record(output_path, duration_seconds=2, sample_rate=16000, channels=1)

    info = sf.info(str(output_path))
    assert info.samplerate == 16000
