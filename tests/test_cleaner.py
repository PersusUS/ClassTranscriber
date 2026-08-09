"""Unit tests for M6 — modules/cleaner.py."""

import sys
from unittest.mock import MagicMock, patch

import pytest

import config
import modules.cleaner as cleaner_mod
from modules.cleaner import _parse_reply, clean_transcript, set_professor_speaker


@pytest.fixture(autouse=True)
def reset_professor():
    """Resets the module-level professor label around each test."""
    cleaner_mod._professor_speaker = None
    yield
    cleaner_mod._professor_speaker = None


def _make_segments():
    """Returns 5 segments: 3 from the professor, 2 from a student."""
    return [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "eh hola a todos"},
        {"start": 5.0, "end": 8.0, "speaker": "SPEAKER_01", "text": "una pregunta"},
        {"start": 8.0, "end": 12.0, "speaker": "SPEAKER_00", "text": "em o sea distribuidos"},
        {"start": 12.0, "end": 15.0, "speaker": "SPEAKER_01", "text": "otra pregunta"},
        {"start": 15.0, "end": 20.0, "speaker": "SPEAKER_00", "text": "el modelo de consistencia"},
    ]


def _reply(text: str):
    """Builds a mock ollama response object."""
    response = MagicMock()
    response.message.content = text
    return response


def _numbered(*lines: str):
    """Formats lines the way the prompt asks the model to answer."""
    return _reply("\n".join(f"{number}| {line}" for number, line in enumerate(lines, 1)))


@patch("modules.cleaner.chat")
def test_clean_calls_ollama(mock_chat):
    """Assert the LLM is actually called."""
    mock_chat.return_value = _numbered("Hola a todos.", "O sea, distribuidos.", "El modelo.")
    set_professor_speaker("SPEAKER_00")

    clean_transcript(_make_segments())

    assert mock_chat.call_count >= 1


@patch("modules.cleaner.chat")
def test_clean_returns_same_structure(mock_chat):
    """Assert the output keeps the input keys."""
    mock_chat.return_value = _numbered("Hola a todos.", "O sea, distribuidos.", "El modelo.")
    set_professor_speaker("SPEAKER_00")

    for segment in clean_transcript(_make_segments()):
        assert {"start", "end", "speaker", "text"} <= set(segment)


@patch("modules.cleaner.chat")
def test_clean_non_professor_unchanged(mock_chat):
    """Segments from other speakers must not be touched."""
    mock_chat.return_value = _numbered("Hola a todos.", "O sea, distribuidos.", "El modelo.")
    set_professor_speaker("SPEAKER_00")

    segments = _make_segments()
    before = [segment["text"] for segment in segments if segment["speaker"] == "SPEAKER_01"]

    result = clean_transcript(segments)
    after = [segment["text"] for segment in result if segment["speaker"] == "SPEAKER_01"]

    assert after == before


@patch("modules.cleaner.chat")
def test_clean_applies_text_in_order(mock_chat):
    """Cleaned lines land on the segment they belong to."""
    mock_chat.return_value = _numbered("Hola a todos.", "O sea, distribuidos.", "El modelo.")
    set_professor_speaker("SPEAKER_00")

    result = clean_transcript(_make_segments())
    professor = [s["text"] for s in result if s["speaker"] == "SPEAKER_00"]

    assert professor == ["Hola a todos.", "O sea, distribuidos.", "El modelo."]


@patch("modules.cleaner.chat")
def test_clean_survives_missing_lines(mock_chat):
    """If the model drops a line, the rest must stay aligned.

    The old positional matching shifted every later line onto the wrong
    timestamp; numbering makes a dropped line a local loss instead.
    """
    mock_chat.return_value = _reply("1| Hola a todos.\n3| El modelo de consistencia.")
    set_professor_speaker("SPEAKER_00")

    result = clean_transcript(_make_segments())
    professor = [s["text"] for s in result if s["speaker"] == "SPEAKER_00"]

    assert professor == [
        "Hola a todos.",
        "em o sea distribuidos",          # untouched, not shifted
        "El modelo de consistencia.",
    ]


@patch("modules.cleaner.chat")
def test_clean_rejects_runaway_output(mock_chat):
    """A model that starts explaining instead of correcting is ignored."""
    mock_chat.return_value = _reply("1| " + "esta línea es una explicación larguísima " * 10)
    set_professor_speaker("SPEAKER_00")

    segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hola"}]
    result = clean_transcript(segments)

    assert result[0]["text"] == "hola"


@patch("modules.cleaner.chat")
def test_clean_keeps_original_text(mock_chat):
    """Rewritten segments keep their previous wording for auditing."""
    mock_chat.return_value = _numbered("Hola a todos.")
    set_professor_speaker("SPEAKER_00")

    segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "eh hola a todos"}]
    result = clean_transcript(segments)

    assert result[0]["original_text"] == "eh hola a todos"


@patch("modules.cleaner.chat")
def test_clean_chunking(mock_chat):
    """5 professor segments with chunk_size=2 means 3 LLM calls."""
    mock_chat.return_value = _numbered("Línea uno.", "Línea dos.")
    set_professor_speaker("SPEAKER_00")

    segments = [
        {"start": float(i), "end": float(i + 1), "speaker": "SPEAKER_00", "text": f"texto {i}"}
        for i in range(5)
    ]

    clean_transcript(segments, chunk_size=2)

    assert mock_chat.call_count == 3


@patch("modules.cleaner.chat")
def test_clean_chunks_by_characters(mock_chat):
    """Long segments split into more chunks even below the count limit."""
    mock_chat.return_value = _numbered("ok")
    set_professor_speaker("SPEAKER_00")

    segments = [
        {"start": float(i), "end": float(i + 1), "speaker": "SPEAKER_00", "text": "x" * 400}
        for i in range(6)
    ]

    clean_transcript(segments, chunk_size=50, chunk_chars=1000)

    assert mock_chat.call_count == 3


@patch("modules.cleaner.chat")
def test_clean_reports_unreachable_ollama(mock_chat):
    """A connection failure produces an actionable error, not a traceback."""
    mock_chat.side_effect = ConnectionError("connection refused")
    set_professor_speaker("SPEAKER_00")

    with pytest.raises(RuntimeError, match="ollama serve"):
        clean_transcript(_make_segments())


@patch("modules.cleaner.chat")
def test_clean_tolerates_other_llm_errors(mock_chat):
    """A model-side failure keeps the original text instead of aborting."""
    mock_chat.side_effect = ValueError("model not found")
    set_professor_speaker("SPEAKER_00")

    result = clean_transcript(_make_segments())

    assert result[0]["text"] == "eh hola a todos"


@patch("modules.cleaner.chat")
def test_clean_professor_argument_overrides_global(mock_chat):
    """The explicit argument wins over the module-level setting."""
    mock_chat.return_value = _numbered("Una pregunta.", "Otra pregunta.")
    set_professor_speaker("SPEAKER_00")

    result = clean_transcript(_make_segments(), professor="SPEAKER_01")

    assert result[0]["text"] == "eh hola a todos"
    assert result[1]["text"] == "Una pregunta."


def test_clean_empty_segments():
    """An empty transcript is returned untouched."""
    assert clean_transcript([]) == []


def test_parse_reply_accepts_several_separators():
    """Models vary the separator; all the common ones are accepted."""
    parsed = _parse_reply("1| uno\n2. dos\n3) tres\n4: cuatro\nbasura", expected=4)

    assert parsed == {1: "uno", 2: "dos", 3: "tres", 4: "cuatro"}


def test_parse_reply_ignores_out_of_range():
    """Line numbers the chunk never sent are discarded."""
    assert _parse_reply("1| uno\n9| nueve", expected=2) == {1: "uno"}


def test_get_chat_binds_to_the_configured_host(monkeypatch):
    """The Ollama client is built lazily and points at OLLAMA_HOST."""
    client_class = MagicMock()
    fake_module = MagicMock()
    fake_module.Client = client_class
    monkeypatch.setitem(sys.modules, "ollama", fake_module)
    monkeypatch.setattr(cleaner_mod, "chat", None)

    cleaner_mod._get_chat()

    assert client_class.call_args.kwargs["host"] == config.OLLAMA_HOST


def test_extract_content_from_object():
    """The modern ollama response object is understood."""
    response = MagicMock()
    response.message.content = "hola"

    assert cleaner_mod._extract_content(response) == "hola"


def test_extract_content_from_dict():
    """Older dict-shaped responses are understood too."""
    assert cleaner_mod._extract_content({"message": {"content": "hola"}}) == "hola"


def test_extract_content_from_garbage():
    """An unrecognised shape yields empty text rather than raising."""
    assert cleaner_mod._extract_content(object()) == ""


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("refused"),
        TimeoutError("timed out"),
        OSError("network unreachable"),
        RuntimeError("failed to connect to ollama"),
    ],
)
def test_is_connection_error_detects_server_down(error):
    """Every flavour of "the server is not there" is recognised."""
    assert cleaner_mod._is_connection_error(error) is True


def test_is_connection_error_ignores_model_errors():
    """A model-side error is not a connection problem."""
    assert cleaner_mod._is_connection_error(ValueError("model not found")) is False


@patch("modules.cleaner.chat")
def test_clean_sends_deterministic_options(mock_chat):
    """Temperature 0: this is correction, not creative writing."""
    mock_chat.return_value = _numbered("Hola.")
    segments = [{"start": 0.0, "end": 1.0, "speaker": "A", "text": "hola"}]

    clean_transcript(segments, professor="A")

    options = mock_chat.call_args.kwargs["options"]
    assert options["temperature"] == 0
    assert options["num_ctx"] == config.OLLAMA_NUM_CTX


@patch("modules.cleaner.chat")
def test_clean_uses_the_spanish_prompt_by_default(mock_chat):
    """A class in Spain gets the Spanish system prompt."""
    mock_chat.return_value = _numbered("Hola.")
    segments = [{"start": 0.0, "end": 1.0, "speaker": "A", "text": "hola"}]

    clean_transcript(segments, professor="A")

    system = mock_chat.call_args.kwargs["messages"][0]["content"]
    assert "transcripciones de clases universitarias en español" in system
    # The old prompt assumed a Chinese-accented English speaker.
    assert "Chinese" not in system


@patch("modules.cleaner.chat")
def test_clean_numbers_the_lines_it_sends(mock_chat):
    """The request itself is numbered — that is what makes realignment safe."""
    mock_chat.return_value = _numbered("Uno.", "Dos.")
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "A", "text": "uno"},
        {"start": 1.0, "end": 2.0, "speaker": "A", "text": "dos"},
    ]

    clean_transcript(segments, professor="A")

    assert mock_chat.call_args.kwargs["messages"][1]["content"] == "1| uno\n2| dos"


@patch("modules.cleaner.chat")
def test_clean_unknown_speaker_leaves_everything_alone(mock_chat, caplog):
    """Pointing the cleanup at a speaker that does not exist is a no-op."""
    with caplog.at_level("WARNING"):
        result = clean_transcript(_make_segments(), professor="SPEAKER_42")

    mock_chat.assert_not_called()
    assert result[0]["text"] == "eh hola a todos"
    assert "No segments found" in caplog.text


@patch("modules.cleaner.chat")
def test_clean_all_speakers_when_no_professor(mock_chat):
    """With no professor set, every segment is cleaned."""
    mock_chat.return_value = _numbered("A.", "B.", "C.", "D.", "E.")

    result = clean_transcript(_make_segments())

    assert [segment["text"] for segment in result] == ["A.", "B.", "C.", "D.", "E."]
