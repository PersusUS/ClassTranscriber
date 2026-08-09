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


@pytest.fixture
def terminal(monkeypatch):
    """Pretends stdin is a terminal, so the interactive prompt is reachable."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)


def test_ask_accepts_a_rank_number(terminal, monkeypatch):
    """The interactive prompt accepts a rank index."""
    monkeypatch.setattr("builtins.input", lambda _: "1")

    assert resolve_professor(_seminar(), interactive=True) == "SPEAKER_01"


def test_ask_defaults_to_the_top_speaker_on_enter(terminal, monkeypatch):
    """Pressing Enter takes the most talkative speaker."""
    monkeypatch.setattr("builtins.input", lambda _: "")

    assert resolve_professor(_seminar(), interactive=True) == "SPEAKER_00"


def test_ask_accepts_none(terminal, monkeypatch):
    """Answering 'none' disables professor filtering."""
    monkeypatch.setattr("builtins.input", lambda _: "none")

    assert resolve_professor(_seminar(), interactive=True) is None


def test_ask_reprompts_after_a_bad_answer(terminal, monkeypatch, capsys):
    """An invalid answer is rejected and the question repeated."""
    answers = iter(["qué?", "SPEAKER_02"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert resolve_professor(_seminar(), interactive=True) == "SPEAKER_02"
    assert "Not a valid choice" in capsys.readouterr().out


def test_ask_handles_closed_stdin(terminal, monkeypatch):
    """A closed stdin falls back to the top speaker instead of crashing."""
    def raise_eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    assert resolve_professor(_seminar(), interactive=True) == "SPEAKER_00"


def test_ask_shows_samples_of_each_voice(terminal, monkeypatch, capsys):
    """The prompt includes sample utterances so the voices can be told apart."""
    monkeypatch.setattr("builtins.input", lambda _: "")

    resolve_professor(_seminar(), interactive=True)

    assert "Parte tres" in capsys.readouterr().out


def test_resolve_professor_matches_label_suffix():
    """A label suffix resolves when the rank index is out of range."""
    segments = [
        {"start": 0.0, "end": 100.0, "speaker": "SPEAKER_02", "text": "mucho"},
        {"start": 100.0, "end": 110.0, "speaker": "SPEAKER_00", "text": "poco"},
    ]

    assert resolve_professor(segments, requested="2", interactive=False) == "SPEAKER_02"


def test_resolve_professor_is_case_insensitive():
    """Speaker labels can be typed in lower case."""
    assert resolve_professor(_lecture(), requested="speaker_01", interactive=False) == "SPEAKER_01"


def test_sample_lines_truncates_long_turns():
    """Samples stay short enough to scan."""
    segments = [{"start": 0.0, "end": 60.0, "speaker": "A", "text": "palabra " * 100}]

    assert sample_lines(segments, "A")[0].endswith("...")


def test_sample_lines_ignores_empty_text():
    """Segments with no text are not offered as samples."""
    segments = [{"start": 0.0, "end": 1.0, "speaker": "A", "text": "   "}]

    assert sample_lines(segments, "A") == []


def test_no_prompt_without_a_terminal(monkeypatch):
    """Piped or unattended runs must never block waiting for input."""
    def fail(_):
        raise AssertionError("input() must not be called without a terminal")

    monkeypatch.setattr("builtins.input", fail)

    assert resolve_professor(_seminar(), interactive=True) == "SPEAKER_00"


def test_resolve_professor_with_no_segments():
    """No speech at all means no professor to identify."""
    assert resolve_professor([], interactive=False) is None
