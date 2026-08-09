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


def test_merge_with_no_word_timings_in_some_segments():
    """A mix of word-timed and plain segments is handled in one pass."""
    diarization = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
    ]
    transcription = [
        {
            "start": 0.0, "end": 2.0, "text": "con palabras",
            "words": [
                {"start": 0.0, "end": 0.8, "word": " con"},
                {"start": 0.8, "end": 1.8, "word": " palabras"},
            ],
        },
        {"start": 2.2, "end": 4.5, "text": "sin palabras"},
    ]

    result = merge(diarization, transcription)

    assert [segment["speaker"] for segment in result] == ["SPEAKER_00", "SPEAKER_01"]
    assert result[1]["text"] == "sin palabras"


def test_merge_smooths_only_isolated_flips():
    """A genuine two-word interjection is preserved, not smoothed away."""
    diarization = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.2, "speaker": "SPEAKER_01"},
        {"start": 2.2, "end": 4.0, "speaker": "SPEAKER_00"},
    ]
    transcription = [
        {
            "start": 0.0, "end": 4.0, "text": "uno dos tres cuatro",
            "words": [
                {"start": 0.0, "end": 0.9, "word": " uno"},
                {"start": 1.1, "end": 1.5, "word": " dos"},
                {"start": 1.6, "end": 2.1, "word": " tres"},
                {"start": 2.4, "end": 3.0, "word": " cuatro"},
            ],
        }
    ]

    result = merge(diarization, transcription)

    assert [segment["speaker"] for segment in result] == [
        "SPEAKER_00", "SPEAKER_01", "SPEAKER_00",
    ]


def test_merge_handles_overlapping_diarization():
    """Overlapping speech (two people at once) still resolves to one label."""
    diarization = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 8.0, "speaker": "SPEAKER_01"},
    ]

    result = merge(diarization, [{"start": 3.5, "end": 7.5, "text": "solapado"}])

    assert result[0]["speaker"] == "SPEAKER_01"


def test_merge_turns_keeps_unknown_separate():
    """UNKNOWN text is not folded into a neighbouring speaker's turn."""
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hola"},
        {"start": 1.1, "end": 2.0, "speaker": "UNKNOWN", "text": "ruido"},
        {"start": 2.1, "end": 3.0, "speaker": "SPEAKER_00", "text": "adiós"},
    ]

    assert len(merge_turns(segments)) == 3


def test_merge_turns_does_not_mutate_the_input():
    """Turn grouping returns new dicts so the cached merge stays intact."""
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "A", "text": "uno"},
        {"start": 1.1, "end": 2.0, "speaker": "A", "text": "dos"},
    ]

    merge_turns(segments)

    assert segments[0]["text"] == "uno"
    assert segments[0]["end"] == 1.0


def test_merge_drops_words_that_become_empty():
    """Whitespace-only word tokens do not create empty segments."""
    diarization = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    transcription = [
        {
            "start": 0.0, "end": 3.0, "text": "hola",
            "words": [
                {"start": 0.0, "end": 0.5, "word": " "},
                {"start": 0.5, "end": 1.0, "word": " hola"},
            ],
        }
    ]

    result = merge(diarization, transcription)

    assert len(result) == 1
    assert result[0]["text"] == "hola"


def test_speaker_index_without_segments():
    """The index answers UNKNOWN rather than failing on empty diarization."""
    from modules.merger import UNKNOWN_SPEAKER, _SpeakerIndex

    assert _SpeakerIndex([]).best_speaker(0.0, 1.0) == (UNKNOWN_SPEAKER, 0.0)
