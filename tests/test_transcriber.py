"""Unit tests for M4 — modules/transcriber.py."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class FakeSegment:
    """Mimics a faster-whisper transcription segment."""

    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    """Mimics the faster-whisper transcription info object."""

    def __init__(self):
        self.language = "en"
        self.language_probability = 0.98


def _build_mock_model():
    """Returns a MagicMock that behaves like WhisperModel."""
    mock_model = MagicMock()
    fake_segments = [
        FakeSegment(0.0, 5.0, "Hello everyone."),
        FakeSegment(5.5, 12.0, "Today we discuss distributed systems."),
    ]
    mock_model.transcribe.return_value = (iter(fake_segments), FakeInfo())
    return mock_model


@patch("modules.transcriber.WhisperModel")
def test_transcribe_returns_list(mock_whisper_cls, tmp_path):
    """Mock WhisperModel. Assert return type is list."""
    import modules.transcriber as mod
    mod._model = None  # reset singleton

    mock_whisper_cls.return_value = _build_mock_model()

    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    from modules.transcriber import transcribe
    result = transcribe(audio_file, language="en")
    assert isinstance(result, list)

    mod._model = None  # cleanup


@patch("modules.transcriber.WhisperModel")
def test_transcribe_segment_keys(mock_whisper_cls, tmp_path):
    """Assert each dict contains 'start', 'end', 'text'."""
    import modules.transcriber as mod
    mod._model = None

    mock_whisper_cls.return_value = _build_mock_model()

    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    from modules.transcriber import transcribe
    result = transcribe(audio_file, language="en")
    for segment in result:
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment

    mod._model = None


def test_transcribe_invalid_path():
    """Assert FileNotFoundError for missing file."""
    from modules.transcriber import transcribe
    bad_path = Path("/nonexistent/audio.wav")
    with pytest.raises(FileNotFoundError):
        transcribe(bad_path)


@patch("modules.transcriber.WhisperModel")
def test_model_singleton(mock_whisper_cls, tmp_path):
    """Call transcribe() twice. Assert WhisperModel constructor is called only once."""
    import modules.transcriber as mod
    mod._model = None

    mock_model = _build_mock_model()
    mock_whisper_cls.return_value = mock_model

    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    from modules.transcriber import transcribe

    # First call — model gets created
    mock_model.transcribe.return_value = (
        iter([FakeSegment(0.0, 5.0, "First call.")]),
        FakeInfo(),
    )
    transcribe(audio_file, language="en")

    # Second call — model should be reused
    mock_model.transcribe.return_value = (
        iter([FakeSegment(0.0, 5.0, "Second call.")]),
        FakeInfo(),
    )
    transcribe(audio_file, language="en")

    assert mock_whisper_cls.call_count == 1

    mod._model = None
