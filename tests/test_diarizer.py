"""Unit tests for M3 — modules/diarizer.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import modules.diarizer as diarizer_mod
from modules.diarizer import _resolve_device, diarize


class FakeSegment:
    """Mimics a pyannote Segment with start/end attributes."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class FakeDiarization:
    """Mimics the pyannote diarization result object."""

    def __init__(self, tracks: list[tuple]):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        return iter(self._tracks)


@pytest.fixture(autouse=True)
def clear_pipeline_cache():
    """Prevents the cached pipeline from leaking between tests."""
    diarizer_mod._pipelines.clear()
    yield
    diarizer_mod._pipelines.clear()


def _pipeline_returning(tracks):
    """Builds a mock pipeline yielding the given tracks."""
    pipeline = MagicMock()
    pipeline.return_value = FakeDiarization(tracks)
    return pipeline


@pytest.fixture
def fake_pipeline():
    """A pipeline that returns three segments from two speakers."""
    return _pipeline_returning(
        [
            (FakeSegment(0.0, 5.0), None, "SPEAKER_00"),
            (FakeSegment(5.5, 12.0), None, "SPEAKER_01"),
            (FakeSegment(12.5, 20.0), None, "SPEAKER_00"),
        ]
    )


@patch("modules.diarizer._load_pipeline")
def test_diarize_returns_list(mock_load, fake_pipeline, tmp_path):
    """Assert the return type is a list."""
    mock_load.return_value = fake_pipeline
    audio = tmp_path / "test.wav"
    audio.touch()

    assert isinstance(diarize(audio, hf_token="fake_token"), list)


@patch("modules.diarizer._load_pipeline")
def test_diarize_segment_keys(mock_load, fake_pipeline, tmp_path):
    """Assert each dict contains 'start', 'end' and 'speaker'."""
    mock_load.return_value = fake_pipeline
    audio = tmp_path / "test.wav"
    audio.touch()

    for segment in diarize(audio, hf_token="fake_token"):
        assert {"start", "end", "speaker"} <= set(segment)


@patch("modules.diarizer._load_pipeline")
def test_diarize_sorted_by_start(mock_load, tmp_path):
    """Assert segments come back sorted by start time."""
    mock_load.return_value = _pipeline_returning(
        [
            (FakeSegment(10.0, 15.0), None, "SPEAKER_01"),
            (FakeSegment(0.0, 5.0), None, "SPEAKER_00"),
            (FakeSegment(5.5, 9.0), None, "SPEAKER_00"),
        ]
    )
    audio = tmp_path / "test.wav"
    audio.touch()

    starts = [segment["start"] for segment in diarize(audio, hf_token="fake_token")]

    assert starts == sorted(starts)


@patch("modules.diarizer._load_pipeline")
def test_diarize_passes_exact_speaker_count(mock_load, fake_pipeline, tmp_path):
    """num_speakers must replace the min/max bounds when given."""
    mock_load.return_value = fake_pipeline
    audio = tmp_path / "test.wav"
    audio.touch()

    diarize(audio, hf_token="fake_token", num_speakers=3)

    kwargs = fake_pipeline.call_args.kwargs
    assert kwargs["num_speakers"] == 3
    assert "max_speakers" not in kwargs


def test_diarize_invalid_path():
    """Assert FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        diarize(Path("/nonexistent/audio.wav"), hf_token="fake_token")


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    """Requesting CUDA on a machine without it must not fail — CPU is fine."""
    monkeypatch.setattr("config.cuda_available", lambda: False)

    assert _resolve_device("cuda") == "cpu"
    assert _resolve_device("auto") == "cpu"
    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_prefers_cuda_when_present(monkeypatch):
    """With CUDA available, 'auto' selects the GPU."""
    monkeypatch.setattr("config.cuda_available", lambda: True)

    assert _resolve_device("auto") == "cuda"
