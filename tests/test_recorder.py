"""Unit tests for M1 — modules/recorder.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import config
from modules.recorder import _report_level, list_input_devices, record


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


def test_report_level_warns_when_too_quiet(caplog):
    """A distant or muted microphone is flagged during the class, not after."""
    with caplog.at_level("WARNING"):
        _report_level(30.0, 16000 * 60, 16000, [-55.0, -57.0], window_peak=-50.0)

    assert "very quiet" in caplog.text


def test_report_level_warns_when_clipping(caplog):
    """Too much input gain is flagged too."""
    with caplog.at_level("WARNING"):
        _report_level(30.0, None, 16000, [-3.0], window_peak=-0.2)

    assert "clipping" in caplog.text


def test_report_level_quiet_when_healthy(caplog):
    """A healthy level produces progress info and no warning."""
    with caplog.at_level("INFO"):
        _report_level(30.0, 16000 * 60, 16000, [-20.0], window_peak=-9.0)

    assert "Recorded 30 / 60 s" in caplog.text
    assert "WARNING" not in caplog.text


def test_report_level_handles_empty_window(caplog):
    """A report with no measurements must not raise."""
    with caplog.at_level("INFO"):
        _report_level(30.0, None, 16000, [], window_peak=float("-inf"))

    assert "Recorded 30 s" in caplog.text


def test_get_sd_caches_the_module(monkeypatch):
    """sounddevice is imported once, lazily, and reused."""
    import modules.recorder as recorder_mod

    fake = MagicMock()
    monkeypatch.setattr(recorder_mod, "sd", None)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    assert recorder_mod._get_sd() is fake
    assert recorder_mod._get_sd() is fake


@patch("modules.recorder.sd")
def test_record_warns_when_nothing_captured(mock_sd, output_dir: Path, caplog):
    """An immediately-empty stream is reported rather than silently accepted."""
    stream = MagicMock()
    stream.read.return_value = (np.zeros((0, 1), dtype=np.float32), False)
    mock_sd.query_devices.return_value = {"name": "Test Microphone"}
    mock_sd.InputStream.return_value.__enter__.return_value = stream

    with caplog.at_level("WARNING"):
        record(output_dir / "empty.wav", duration_seconds=5)

    assert "No audio was captured" in caplog.text


@patch("modules.recorder.sd")
def test_record_warns_on_overflow(mock_sd, output_dir: Path, caplog):
    """A dropped buffer is surfaced: it means missing lecture audio."""
    stream = MagicMock()
    stream.read.side_effect = lambda frames: (np.zeros((frames, 1), dtype=np.float32), True)
    mock_sd.query_devices.return_value = {"name": "Test Microphone"}
    mock_sd.InputStream.return_value.__enter__.return_value = stream

    with caplog.at_level("WARNING"):
        record(output_dir / "over.wav", duration_seconds=1)

    assert "overflow" in caplog.text


@patch("modules.recorder.sd")
def test_record_reports_level_periodically(mock_sd, output_dir: Path, caplog, monkeypatch):
    """The level is reported as the class goes on, not only at the end."""
    monkeypatch.setattr(config, "LEVEL_LOG_INTERVAL", 2)
    _wire(mock_sd)

    with caplog.at_level("INFO"):
        record(output_dir / "long.wav", duration_seconds=6, sample_rate=16000)

    reports = [line for line in caplog.text.splitlines() if "level" in line]
    assert len(reports) >= 3


@patch("modules.recorder.sd")
def test_record_stops_when_the_event_is_set(mock_sd, output_dir: Path):
    """A Stop button ends an open-ended recording and keeps the audio.

    A GUI cannot deliver a KeyboardInterrupt to a worker thread, so the
    event is the only way to end a recording with no duration set.
    """
    import threading

    stop = threading.Event()
    stream = MagicMock()
    reads = {"count": 0}

    def read(frames):
        reads["count"] += 1
        if reads["count"] >= 3:
            stop.set()
        return np.zeros((frames, 1), dtype=np.float32), False

    stream.read.side_effect = read
    mock_sd.query_devices.return_value = {"name": "Test Microphone"}
    mock_sd.InputStream.return_value.__enter__.return_value = stream

    output_path = output_dir / "stopped.wav"
    record(output_path, duration_seconds=None, sample_rate=16000, stop_event=stop)

    assert sf.info(str(output_path)).frames == 3 * 16000


@patch("modules.recorder.sd")
def test_record_checks_the_event_before_reading(mock_sd, output_dir: Path):
    """An event already set means nothing is recorded at all."""
    import threading

    stop = threading.Event()
    stop.set()
    _wire(mock_sd)

    output_path = output_dir / "none.wav"
    record(output_path, duration_seconds=60, stop_event=stop)

    assert sf.info(str(output_path)).frames == 0


@patch("modules.recorder.sd")
def test_record_reports_levels_to_the_callback(mock_sd, output_dir: Path):
    """The live meter gets one reading per block, with elapsed time."""
    _wire(mock_sd)
    readings = []

    record(
        output_dir / "levels.wav",
        duration_seconds=3,
        sample_rate=16000,
        on_level=lambda elapsed, level, peak: readings.append((elapsed, level, peak)),
    )

    assert len(readings) == 3
    assert [round(reading[0]) for reading in readings] == [1, 2, 3]


@patch("modules.recorder.sd")
def test_record_survives_a_broken_level_callback(mock_sd, output_dir: Path, caplog):
    """A crashing meter must never cost the user the class."""
    _wire(mock_sd)

    def explode(elapsed, level, peak):
        raise ValueError("widget destruido")

    output_path = output_dir / "resilient.wav"
    with caplog.at_level("ERROR"):
        record(output_path, duration_seconds=2, sample_rate=16000, on_level=explode)

    assert sf.info(str(output_path)).frames == 2 * 16000
    assert "Level callback failed" in caplog.text
