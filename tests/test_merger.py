"""Unit tests for M5 — modules/merger.py."""

import pytest

from modules.merger import merge


def test_merge_basic_overlap():
    """Simple 2-segment case. Assert speaker is correctly assigned."""
    diarization = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 12.0, "speaker": "SPEAKER_01"},
    ]
    transcription = [
        {"start": 0.5, "end": 4.5, "text": "Hello everyone."},
        {"start": 5.5, "end": 11.0, "text": "Today we discuss systems."},
    ]

    result = merge(diarization, transcription)

    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_merge_no_overlap():
    """Transcription segment with no overlapping diarization. Assert speaker is UNKNOWN."""
    diarization = [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
    ]
    transcription = [
        {"start": 10.0, "end": 15.0, "text": "No overlap here."},
    ]

    result = merge(diarization, transcription)

    assert result[0]["speaker"] == "UNKNOWN"


def test_merge_output_keys():
    """Assert each dict contains 'start', 'end', 'speaker', 'text'."""
    diarization = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
    ]
    transcription = [
        {"start": 0.0, "end": 4.0, "text": "Some text."},
    ]

    result = merge(diarization, transcription)

    for segment in result:
        assert "start" in segment
        assert "end" in segment
        assert "speaker" in segment
        assert "text" in segment


def test_merge_sorted_output():
    """Assert output is sorted by 'start'."""
    diarization = [
        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
    ]
    transcription = [
        {"start": 8.0, "end": 9.0, "text": "Third."},
        {"start": 0.0, "end": 3.0, "text": "First."},
        {"start": 4.0, "end": 6.0, "text": "Second."},
    ]

    result = merge(diarization, transcription)
    starts = [s["start"] for s in result]
    assert starts == sorted(starts)


def test_merge_empty_inputs():
    """Assert raises ValueError for empty input lists."""
    with pytest.raises(ValueError):
        merge([], [{"start": 0.0, "end": 1.0, "text": "text"}])

    with pytest.raises(ValueError):
        merge([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}], [])
