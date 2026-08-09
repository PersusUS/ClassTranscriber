"""Unit tests for modules/speaker_id.py."""

import pytest

from modules.speaker_id import (
    format_stats,
    resolve_professor,
    sample_lines,
    speaker_stats,
)


def _lecture():
    """A lecture where the professor clearly dominates the airtime."""
    return [
        {"start": 0.0, "end": 300.0, "speaker": "SPEAKER_00", "text": "Explicación larga " * 5},
        {"start": 300.0, "end": 320.0, "speaker": "SPEAKER_01", "text": "Una pregunta corta"},
        {"start": 320.0, "end": 600.0, "speaker": "SPEAKER_00", "text": "Más explicación"},
    ]


def _seminar():
    """A seminar where nobody dominates — the ambiguous case."""
    return [
        {"start": 0.0, "end": 100.0, "speaker": "SPEAKER_00", "text": "Parte uno"},
        {"start": 100.0, "end": 190.0, "speaker": "SPEAKER_01", "text": "Parte dos"},
        {"start": 190.0, "end": 280.0, "speaker": "SPEAKER_02", "text": "Parte tres"},
    ]


def test_speaker_stats_ranks_by_time():
    """The most talkative speaker comes first, with a correct share."""
    stats = speaker_stats(_lecture())

    assert stats[0]["speaker"] == "SPEAKER_00"
    assert stats[0]["seconds"] == pytest.approx(580.0)
    assert stats[0]["share"] == pytest.approx(580 / 600, rel=1e-3)
    assert stats[0]["turns"] == 2


def test_speaker_stats_counts_words():
    """Word counts come from the transcript text."""
    stats = speaker_stats([{"start": 0, "end": 1, "speaker": "A", "text": "una dos tres"}])

    assert stats[0]["words"] == 3


def test_speaker_stats_empty():
    """No segments means no statistics."""
    assert speaker_stats([]) == []


def test_resolve_professor_picks_dominant_speaker():
    """In a normal lecture the professor is found without asking."""
    assert resolve_professor(_lecture(), interactive=False) == "SPEAKER_00"


def test_resolve_professor_explicit_label():
    """An explicit label is honoured verbatim."""
    assert resolve_professor(_lecture(), requested="SPEAKER_01", interactive=False) == "SPEAKER_01"


def test_resolve_professor_by_rank_index():
    """A rank index selects from the airtime ranking."""
    assert resolve_professor(_lecture(), requested="1", interactive=False) == "SPEAKER_01"


def test_resolve_professor_none_disables_filtering():
    """'none' keeps every speaker in the output."""
    assert resolve_professor(_lecture(), requested="none", interactive=False) is None


def test_resolve_professor_unknown_label():
    """An unknown speaker is a user error, not a silent fallback."""
    with pytest.raises(ValueError):
        resolve_professor(_lecture(), requested="SPEAKER_42", interactive=False)


def test_resolve_professor_non_interactive_falls_back(caplog):
    """Without a terminal, the ambiguous case still returns a usable answer."""
    with caplog.at_level("WARNING"):
        professor = resolve_professor(_seminar(), interactive=False)

    assert professor == "SPEAKER_00"
    assert any("speaking time" in record.message for record in caplog.records)


def test_resolve_professor_auto_skips_the_prompt(monkeypatch):
    """'auto' takes the top speaker even when the margin is small."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)

    assert resolve_professor(_seminar(), requested="auto", interactive=True) == "SPEAKER_00"


def test_resolve_professor_single_speaker():
    """One speaker is always the professor, no matter the share."""
    segments = [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00", "text": "solo"}]

    assert resolve_professor(segments, interactive=False) == "SPEAKER_00"


def test_sample_lines_returns_longest_utterances():
    """The samples shown to the user are the most recognisable ones."""
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "A", "text": "sí"},
        {"start": 1.0, "end": 5.0, "speaker": "A", "text": "una explicación bastante más larga"},
        {"start": 5.0, "end": 6.0, "speaker": "B", "text": "otra voz"},
    ]

    lines = sample_lines(segments, "A", count=1)

    assert "explicación bastante más larga" in lines[0]
    assert lines[0].startswith("[00:00:01]")


def test_format_stats_is_tabular():
    """The breakdown renders one header row plus one row per speaker."""
    rendered = format_stats(speaker_stats(_lecture()))

    assert rendered.splitlines()[0].startswith("#")
    assert len(rendered.splitlines()) == 3
