"""Unit tests for M6 — modules/cleaner.py."""

from unittest.mock import patch, MagicMock

import pytest

import modules.cleaner as cleaner_mod
from modules.cleaner import clean_transcript, set_professor_speaker


@pytest.fixture(autouse=True)
def reset_professor():
    """Reset the professor speaker before each test."""
    cleaner_mod._professor_speaker = None
    yield
    cleaner_mod._professor_speaker = None


def _make_segments():
    """Helper: returns 5 segments, 3 from professor and 2 from student."""
    return [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "uh hello everyone"},
        {"start": 5.0, "end": 8.0, "speaker": "SPEAKER_01", "text": "question here"},
        {"start": 8.0, "end": 12.0, "speaker": "SPEAKER_00", "text": "um so like distributed"},
        {"start": 12.0, "end": 15.0, "speaker": "SPEAKER_01", "text": "another question"},
        {"start": 15.0, "end": 20.0, "speaker": "SPEAKER_00", "text": "er the consistency model"},
    ]


def _mock_chat_response(text: str):
    """Builds a mock response object for ollama.chat."""
    resp = MagicMock()
    resp.message.content = text
    return resp


@patch("modules.cleaner.chat")
def test_clean_calls_ollama(mock_chat):
    """Mock ollama.chat. Assert it is called at least once."""
    mock_chat.return_value = _mock_chat_response("Hello everyone.\nSo distributed.\nThe consistency model.")
    set_professor_speaker("SPEAKER_00")

    segments = _make_segments()
    clean_transcript(segments, chunk_size=50)

    assert mock_chat.call_count >= 1


@patch("modules.cleaner.chat")
def test_clean_returns_same_structure(mock_chat):
    """Assert output has the same keys as input."""
    mock_chat.return_value = _mock_chat_response("Hello everyone.\nSo distributed.\nThe consistency model.")
    set_professor_speaker("SPEAKER_00")

    segments = _make_segments()
    result = clean_transcript(segments, chunk_size=50)

    for segment in result:
        assert "start" in segment
        assert "end" in segment
        assert "speaker" in segment
        assert "text" in segment


@patch("modules.cleaner.chat")
def test_clean_non_professor_unchanged(mock_chat):
    """Segments from non-professor speakers must not be modified."""
    mock_chat.return_value = _mock_chat_response("Hello everyone.\nSo distributed.\nThe consistency model.")
    set_professor_speaker("SPEAKER_00")

    segments = _make_segments()
    original_student_texts = [
        seg["text"] for seg in segments if seg["speaker"] == "SPEAKER_01"
    ]

    result = clean_transcript(segments, chunk_size=50)
    student_texts = [
        seg["text"] for seg in result if seg["speaker"] == "SPEAKER_01"
    ]

    assert student_texts == original_student_texts


@patch("modules.cleaner.chat")
def test_clean_chunking(mock_chat):
    """With chunk_size=2 and 5 professor segments, assert Ollama is called 3 times."""
    mock_chat.return_value = _mock_chat_response("Line one.\nLine two.")
    set_professor_speaker("SPEAKER_00")

    # 5 professor segments
    segments = [
        {"start": float(i), "end": float(i + 1), "speaker": "SPEAKER_00", "text": f"text {i}"}
        for i in range(5)
    ]

    clean_transcript(segments, chunk_size=2)

    assert mock_chat.call_count == 3
