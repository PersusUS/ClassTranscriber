"""Unit tests for M5 — modules/merger.py."""

import pytest

from modules.merger import merge, merge_turns


def test_merge_basic_overlap():
    """Simple 2-segment case. Assert the speaker is correctly assigned."""
    diarization = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 12.0, "speaker": "SPEAKER_01"},
    ]
    transcription = [
        {"start": 0.5, "end": 4.5, "text": "Hola a todos."},
        {"start": 5.5, "end": 11.0, "text": "Hoy vemos sistemas."},
    ]

    result = merge(diarization, transcription)

    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_merge_no_overlap():
    """A segment with no overlapping diarization is labelled UNKNOWN."""
    result = merge(
        [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}],
        [{"start": 10.0, "end": 15.0, "text": "Sin solape."}],
    )

    assert result[0]["speaker"] == "UNKNOWN"


def test_merge_output_keys():
    """Assert each dict contains 'start', 'end', 'speaker' and 'text'."""
    result = merge(
        [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}],
        [{"start": 0.0, "end": 4.0, "text": "Algo de texto."}],
    )

    for segment in result:
        assert {"start", "end", "speaker", "text"} <= set(segment)


def test_merge_sorted_output():
    """Assert output is sorted by start time."""
    result = merge(
        [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}],
        [
            {"start": 8.0, "end": 9.0, "text": "Tercero."},
            {"start": 0.0, "end": 3.0, "text": "Primero."},
            {"start": 4.0, "end": 6.0, "text": "Segundo."},
        ],
    )

    starts = [segment["start"] for segment in result]
    assert starts == sorted(starts)


def test_merge_empty_inputs():
    """Assert ValueError for empty input lists."""
    with pytest.raises(ValueError):
        merge([], [{"start": 0.0, "end": 1.0, "text": "text"}])

    with pytest.raises(ValueError):
        merge([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}], [])


def test_merge_picks_maximum_overlap():
    """When two speakers overlap a segment, the dominant one wins."""
    diarization = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"},
        {"start": 1.0, "end": 10.0, "speaker": "SPEAKER_00"},
    ]

    result = merge(diarization, [{"start": 0.0, "end": 10.0, "text": "Texto largo."}])

    assert result[0]["speaker"] == "SPEAKER_00"


def test_merge_splits_segment_at_speaker_change():
    """A Whisper segment spanning two speakers must be split, not attributed whole.

    This is what keeps a student's question out of the professor's transcript.
    """
    diarization = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    transcription = [
        {
            "start": 0.0,
            "end": 4.0,
            "text": "Esto es continuo ¿alguna duda?",
            "words": [
                {"start": 0.0, "end": 0.5, "word": " Esto"},
                {"start": 0.5, "end": 1.0, "word": " es"},
                {"start": 1.0, "end": 1.9, "word": " continuo"},
                {"start": 2.1, "end": 2.6, "word": " alguna"},
                {"start": 2.6, "end": 3.4, "word": " duda"},
            ],
        }
    ]

    result = merge(diarization, transcription)

    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["text"] == "Esto es continuo"
    assert result[1]["speaker"] == "SPEAKER_01"
    assert result[1]["text"] == "alguna duda"


def test_merge_smooths_isolated_word_flip():
    """A single word flipped mid-sentence is diarization jitter, not a turn."""
    diarization = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 1.4, "speaker": "SPEAKER_01"},
        {"start": 1.4, "end": 3.0, "speaker": "SPEAKER_00"},
    ]
    transcription = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "la derivada parcial vale cero",
            "words": [
                {"start": 0.0, "end": 0.4, "word": " la"},
                {"start": 0.4, "end": 0.9, "word": " derivada"},
                {"start": 1.05, "end": 1.35, "word": " parcial"},
                {"start": 1.5, "end": 2.0, "word": " vale"},
                {"start": 2.0, "end": 2.5, "word": " cero"},
            ],
        }
    ]

    result = merge(diarization, transcription)

    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["text"] == "la derivada parcial vale cero"


def test_merge_can_disable_word_level():
    """use_words=False keeps the old whole-segment behaviour."""
    diarization = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    transcription = [
        {
            "start": 0.0,
            "end": 4.0,
            "text": "Todo junto",
            "words": [
                {"start": 0.0, "end": 1.5, "word": " Todo"},
                {"start": 2.5, "end": 3.5, "word": " junto"},
            ],
        }
    ]

    result = merge(diarization, transcription, use_words=False)

    assert len(result) == 1


def test_merge_turns_joins_same_speaker():
    """Consecutive fragments from one speaker become a single turn."""
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Primera parte"},
        {"start": 2.3, "end": 4.0, "speaker": "SPEAKER_00", "text": "segunda parte"},
        {"start": 4.5, "end": 6.0, "speaker": "SPEAKER_01", "text": "una pregunta"},
    ]

    turns = merge_turns(segments)

    assert len(turns) == 2
    assert turns[0]["text"] == "Primera parte segunda parte"
    assert turns[0]["end"] == 4.0


def test_merge_turns_respects_gap_and_length():
    """A long silence, or an over-long turn, starts a new one."""
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Antes"},
        {"start": 30.0, "end": 32.0, "speaker": "SPEAKER_00", "text": "Después"},
    ]

    assert len(merge_turns(segments, max_gap=1.5)) == 2
    assert len(merge_turns(segments, max_gap=60.0, max_seconds=10.0)) == 2


def test_merge_turns_empty():
    """An empty list merges to an empty list."""
    assert merge_turns([]) == []
