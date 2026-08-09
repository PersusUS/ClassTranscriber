"""Unit tests for M1 — modules/recorder.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from modules.recorder import list_input_devices, record


def _fake_stream(sample_rate: int = 16000, channels: int = 1):
    """Builds a mock InputStream that returns silence of the requested size."""
    stream = MagicMock()
    stream.read.side_effect = lambda frames: (
        np.zeros((frames, channels), dtype=np.float32),
        False,
    )
    return stream


def _wire(mock_sd, channels: int = 1):
    """Wires a MagicMock sounddevice module with a working InputStream."""
    mock_sd.query_devices.return_value = {"name": "Test Microphone"}
    mock_sd.InputStream.return_value.__enter__.return_value = _fake_stream(channels=channels)
    return mock_sd


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Provides a temporary directory for test output files."""
    return tmp_path


@patch("modules.recorder.sd")
def test_record_creates_file(mock_sd, output_dir: Path):
    """Assert the output WAV file is created."""
    _wire(mock_sd)
    output_path = output_dir / "test.wav"

    record(output_path, duration_seconds=2, sample_rate=16000, channels=1)

    assert output_path.exists()


@patch("modules.recorder.sd")
def test_record_invalid_directory(mock_sd):
    """Assert FileNotFoundError when the parent directory is missing."""
    _wire(mock_sd)

    with pytest.raises(FileNotFoundError):
        record(Path("/nonexistent/directory/test.wav"), duration_seconds=2)


@patch("modules.recorder.sd")
def test_record_returns_path(mock_sd, output_dir: Path):
    """Assert the return value equals the given output_path."""
    _wire(mock_sd)
    output_path = output_dir / "test.wav"

    assert record(output_path, duration_seconds=2) == output_path


@patch("modules.recorder.sd")
def test_record_correct_samplerate(mock_sd, output_dir: Path):
    """Assert the written file carries the requested sample rate."""
    _wire(mock_sd)
    output_path = output_dir / "test.wav"

    record(output_path, duration_seconds=2, sample_rate=16000, channels=1)

    assert sf.info(str(output_path)).samplerate == 16000


@patch("modules.recorder.sd")
def test_record_writes_exact_duration(mock_sd, output_dir: Path):
    """Assert the recording stops at the requested duration, not past it."""
    _wire(mock_sd)
    output_path = output_dir / "test.wav"

    record(output_path, duration_seconds=3, sample_rate=16000)

    assert sf.info(str(output_path)).frames == 3 * 16000


@patch("modules.recorder.sd")
def test_record_keeps_audio_on_interrupt(mock_sd, output_dir: Path):
    """A Ctrl+C mid-class must keep everything recorded so far."""
    stream = MagicMock()
    calls = {"count": 0}

    def read(frames):
        calls["count"] += 1
        if calls["count"] > 2:
            raise KeyboardInterrupt
        return np.zeros((frames, 1), dtype=np.float32), False

    stream.read.side_effect = read
    mock_sd.query_devices.return_value = {"name": "Test Microphone"}
    mock_sd.InputStream.return_value.__enter__.return_value = stream

    output_path = output_dir / "test.wav"
    result = record(output_path, duration_seconds=3600, sample_rate=16000)

    assert result == output_path
    assert sf.info(str(output_path)).frames == 2 * 16000


@patch("modules.recorder.sd")
def test_record_no_input_device(mock_sd, output_dir: Path):
    """Assert RuntimeError when no input device can be queried."""
    mock_sd.query_devices.side_effect = OSError("no device")

    with pytest.raises(RuntimeError):
        record(output_dir / "test.wav", duration_seconds=1)


@patch("modules.recorder.sd")
def test_list_input_devices_filters_outputs(mock_sd):
    """Only devices with input channels are listed."""
    mock_sd.query_devices.return_value = [
        {"name": "Speakers", "max_input_channels": 0},
        {"name": "Laptop mic", "max_input_channels": 2},
    ]
    mock_sd.default.device = [1, 0]

    devices = list_input_devices()

    assert [device["name"] for device in devices] == ["Laptop mic"]
    assert devices[0]["default"] is True
