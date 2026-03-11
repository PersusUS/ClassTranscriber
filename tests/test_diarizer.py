"""Unit tests for M3 — modules/diarizer.py."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from modules.diarizer import diarize


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


@pytest.fixture
def fake_pipeline():
    """Builds a mock pipeline that returns two speaker segments."""
    tracks = [
        (FakeSegment(0.0, 5.0), None, "SPEAKER_00"),
        (FakeSegment(5.5, 12.0), None, "SPEAKER_01"),
        (FakeSegment(12.5, 20.0), None, "SPEAKER_00"),
    ]
    diarization_result = FakeDiarization(tracks)

    pipeline_instance = MagicMock()
    pipeline_instance.return_value = diarization_result
    return pipeline_instance


@patch("modules.diarizer.torch")
@patch("modules.diarizer.Pipeline")
def test_diarize_returns_list(mock_pipeline_cls, mock_torch, fake_pipeline, tmp_path):
    """Mock the pipeline. Assert return type is list."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.device.return_value = "cuda"
    mock_pipeline_cls.from_pretrained.return_value = fake_pipeline

    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    result = diarize(audio_file, hf_token="fake_token")
    assert isinstance(result, list)


@patch("modules.diarizer.torch")
@patch("modules.diarizer.Pipeline")
def test_diarize_segment_keys(mock_pipeline_cls, mock_torch, fake_pipeline, tmp_path):
    """Assert each dict in the result contains keys 'start', 'end', 'speaker'."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.device.return_value = "cuda"
    mock_pipeline_cls.from_pretrained.return_value = fake_pipeline

    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    result = diarize(audio_file, hf_token="fake_token")
    for segment in result:
        assert "start" in segment
        assert "end" in segment
        assert "speaker" in segment


@patch("modules.diarizer.torch")
@patch("modules.diarizer.Pipeline")
def test_diarize_sorted_by_start(mock_pipeline_cls, mock_torch, tmp_path):
    """Assert segments are sorted by 'start' ascending."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.device.return_value = "cuda"

    # Provide tracks in unsorted order
    tracks = [
        (FakeSegment(10.0, 15.0), None, "SPEAKER_01"),
        (FakeSegment(0.0, 5.0), None, "SPEAKER_00"),
        (FakeSegment(5.5, 9.0), None, "SPEAKER_00"),
    ]
    pipeline_instance = MagicMock()
    pipeline_instance.return_value = FakeDiarization(tracks)
    mock_pipeline_cls.from_pretrained.return_value = pipeline_instance

    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    result = diarize(audio_file, hf_token="fake_token")
    starts = [s["start"] for s in result]
    assert starts == sorted(starts)


def test_diarize_invalid_path():
    """Assert FileNotFoundError is raised for missing file."""
    bad_path = Path("/nonexistent/audio.wav")
    with pytest.raises(FileNotFoundError):
        diarize(bad_path, hf_token="fake_token")
