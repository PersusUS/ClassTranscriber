"""Unit tests for M4 — modules/transcriber.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config
import modules.transcriber as transcriber_mod
from modules.transcriber import _filter_segments, is_hallucination, transcribe


class FakeWord:
    """Mimics a faster-whisper word with timings."""

    def __init__(self, start: float, end: float, word: str):
        self.start = start
        self.end = end
        self.word = word


class FakeSegment:
    """Mimics a faster-whisper transcription segment."""

    def __init__(self, start, end, text, avg_logprob=-0.2, no_speech_prob=0.05, words=None):
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.words = words


class FakeInfo:
    """Mimics the faster-whisper transcription info object."""

    language = "es"
    language_probability = 0.98
    duration = 12.0


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Keeps the module-level model cache from leaking between tests."""
    transcriber_mod._models.clear()
    yield
    transcriber_mod._models.clear()


def _mock_model(segments=None):
    """Returns a MagicMock behaving like WhisperModel."""
    segments = segments if segments is not None else [
        FakeSegment(0.0, 5.0, "Buenos días a todos."),
        FakeSegment(5.5, 12.0, "Hoy vemos sistemas distribuidos."),
    ]
    model = MagicMock()
    model.transcribe.return_value = (iter(segments), FakeInfo())
    return model


@patch("modules.transcriber._load_model")
def test_transcribe_returns_list(mock_load, tmp_path):
    """Assert the return type is a list of dicts."""
    mock_load.return_value = _mock_model()
    audio = tmp_path / "test.wav"
    audio.touch()

    result = transcribe(audio)

    assert isinstance(result, list)
    assert len(result) == 2


@patch("modules.transcriber._load_model")
def test_transcribe_segment_keys(mock_load, tmp_path):
    """Assert each segment carries timings, text and confidence data."""
    mock_load.return_value = _mock_model()
    audio = tmp_path / "test.wav"
    audio.touch()

    for segment in transcribe(audio):
        assert {"start", "end", "text", "avg_logprob", "no_speech_prob"} <= set(segment)


@patch("modules.transcriber._load_model")
def test_transcribe_keeps_word_timestamps(mock_load, tmp_path):
    """Word timings must survive into the output — the merger needs them."""
    words = [FakeWord(0.0, 0.4, " Buenos"), FakeWord(0.4, 0.9, " días")]
    mock_load.return_value = _mock_model([FakeSegment(0.0, 1.0, "Buenos días", words=words)])
    audio = tmp_path / "test.wav"
    audio.touch()

    result = transcribe(audio)

    assert result[0]["words"][1]["word"] == " días"


@patch("modules.transcriber._load_model")
def test_transcribe_uses_spanish_by_default(mock_load, tmp_path):
    """The default language must be Spanish, with the lecture prompt attached."""
    model = _mock_model()
    mock_load.return_value = model
    audio = tmp_path / "test.wav"
    audio.touch()

    transcribe(audio)

    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["language"] == "es"
    assert "clase universitaria" in kwargs["initial_prompt"]
    # Feeding Whisper its own output is what causes repetition loops in noise.
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["vad_filter"] is True


def test_transcribe_invalid_path():
    """Assert FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        transcribe(Path("/nonexistent/audio.wav"))


def test_model_is_cached(tmp_path, monkeypatch):
    """Two transcriptions with the same settings load the weights once."""
    constructor = MagicMock(return_value=_mock_model())
    fake_module = MagicMock()
    fake_module.WhisperModel = constructor
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    audio = tmp_path / "test.wav"
    audio.touch()

    settings = config.resolve_settings(profile="low", device="cpu")
    transcribe(audio, settings=settings)
    constructor.return_value.transcribe.return_value = (iter([]), FakeInfo())
    transcribe(audio, settings=settings)

    assert constructor.call_count == 1


def test_model_falls_back_to_cpu(tmp_path, monkeypatch):
    """A GPU that cannot load the model must fall back instead of failing."""
    attempts = []

    def constructor(model, device, compute_type, cpu_threads):
        attempts.append((device, compute_type))
        if device == "cuda":
            raise RuntimeError("CUDA failed to initialize")
        return _mock_model()

    fake_module = MagicMock()
    fake_module.WhisperModel = constructor
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    audio = tmp_path / "test.wav"
    audio.touch()

    settings = config.resolve_settings(profile="low", device="cpu")
    settings.device = "cuda"        # Force the CUDA attempt
    settings.compute_type = "float16"

    transcribe(audio, settings=settings)

    assert attempts[0][0] == "cuda"
    assert attempts[-1] == ("cpu", "int8")


def test_is_hallucination_detects_amara():
    """The classic Spanish silence hallucination must be recognised."""
    assert is_hallucination("Subtítulos realizados por la comunidad de Amara.org")
    assert is_hallucination("   ")
    assert not is_hallucination("La transformada de Fourier convierte tiempo en frecuencia.")


def test_filter_segments_drops_noise():
    """Hallucinations, low-confidence output and loops are all removed."""
    segments = [
        {"text": "Contenido válido.", "avg_logprob": -0.3, "no_speech_prob": 0.1},
        {"text": "Subtítulos realizados por la comunidad de Amara.org",
         "avg_logprob": -0.2, "no_speech_prob": 0.1},
        {"text": "Ruido ininteligible", "avg_logprob": -2.5, "no_speech_prob": 0.1},
        {"text": "Silencio", "avg_logprob": -0.2, "no_speech_prob": 0.99},
    ]

    kept = _filter_segments(segments)

    assert [segment["text"] for segment in kept] == ["Contenido válido."]


def test_filter_segments_keeps_first_repetition():
    """A decoding loop keeps one copy, not all of them."""
    segments = [
        {"text": "y ya está", "avg_logprob": -0.3, "no_speech_prob": 0.1}
        for _ in range(5)
    ]

    kept = _filter_segments(segments)

    assert len(kept) == 2      # The first plus one repeat before the loop is detected


@patch("modules.transcriber._load_model")
def test_transcribe_logs_progress_with_eta(mock_load, tmp_path, caplog):
    """Long CPU runs report progress; a silent terminal looks like a hang."""
    segments = [FakeSegment(float(i * 30), float(i * 30 + 30), f"frase {i}") for i in range(5)]
    mock_load.return_value = _mock_model(segments)
    audio = tmp_path / "test.wav"
    audio.touch()

    with caplog.at_level("INFO"):
        transcribe(audio)

    assert "realtime" in caplog.text


@patch("modules.transcriber._load_model")
def test_transcribe_language_argument_overrides_settings(mock_load, tmp_path):
    """An explicit language beats the configured default."""
    model = _mock_model()
    mock_load.return_value = model
    audio = tmp_path / "test.wav"
    audio.touch()

    transcribe(audio, language="EN")

    assert model.transcribe.call_args.kwargs["language"] == "en"


@patch("modules.transcriber._load_model")
def test_transcribe_custom_initial_prompt(mock_load, tmp_path):
    """A subject-specific prompt can be supplied to bias vocabulary."""
    model = _mock_model()
    mock_load.return_value = model
    audio = tmp_path / "test.wav"
    audio.touch()

    transcribe(audio, initial_prompt="Álgebra lineal, autovalores, matrices.")

    assert "autovalores" in model.transcribe.call_args.kwargs["initial_prompt"]


@patch("modules.transcriber._load_model")
def test_transcribe_can_disable_word_timestamps(mock_load, tmp_path):
    """Word timings can be turned off to save a little CPU."""
    model = _mock_model()
    mock_load.return_value = model
    audio = tmp_path / "test.wav"
    audio.touch()

    result = transcribe(audio, word_timestamps=False)

    assert model.transcribe.call_args.kwargs["word_timestamps"] is False
    assert "words" not in result[0]


@patch("modules.transcriber._load_model")
def test_transcribe_applies_vad_parameters(mock_load, tmp_path):
    """VAD is tuned for noisy rooms, not left at library defaults."""
    model = _mock_model()
    mock_load.return_value = model
    audio = tmp_path / "test.wav"
    audio.touch()

    transcribe(audio)

    vad = model.transcribe.call_args.kwargs["vad_parameters"]
    assert vad["min_silence_duration_ms"] == config.VAD_MIN_SILENCE_MS
    assert vad["speech_pad_ms"] == config.VAD_SPEECH_PAD_MS


@patch("modules.transcriber._load_model")
def test_transcribe_drops_words_without_timings(mock_load, tmp_path):
    """Words faster-whisper could not time are skipped, not crashed on."""
    words = [FakeWord(0.0, 0.4, " Buenos"), FakeWord(None, None, " días")]
    mock_load.return_value = _mock_model([FakeSegment(0.0, 1.0, "Buenos días", words=words)])
    audio = tmp_path / "test.wav"
    audio.touch()

    result = transcribe(audio)

    assert [word["word"] for word in result[0]["words"]] == [" Buenos"]


def test_model_load_failure_is_reported(tmp_path, monkeypatch):
    """If no configuration loads, the error names the model."""
    fake_module = MagicMock()
    fake_module.WhisperModel = MagicMock(side_effect=RuntimeError("nope"))
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    audio = tmp_path / "test.wav"
    audio.touch()

    with pytest.raises(RuntimeError, match="Could not load Whisper model"):
        transcribe(audio, settings=config.resolve_settings(profile="low", device="cpu"))


def test_filter_segments_keeps_confident_speech():
    """Nothing is dropped when the audio is clean."""
    segments = [
        {"text": "Primera frase.", "avg_logprob": -0.2, "no_speech_prob": 0.02},
        {"text": "Segunda frase.", "avg_logprob": -0.4, "no_speech_prob": 0.05},
    ]

    assert len(_filter_segments(segments)) == 2


def test_filter_segments_tolerates_missing_confidence():
    """Segments without confidence data are kept rather than discarded."""
    segments = [{"text": "Frase.", "avg_logprob": None, "no_speech_prob": None}]

    assert len(_filter_segments(segments)) == 1


@patch("modules.transcriber._load_model")
def test_transcribe_progress_without_known_duration(mock_load, tmp_path, caplog):
    """Progress is still reported when the audio duration is unknown."""
    class NoDurationInfo:
        language = "es"
        language_probability = 0.9
        duration = None

    model = MagicMock()
    model.transcribe.return_value = (
        iter([FakeSegment(0.0, 90.0, "una frase larga")]),
        NoDurationInfo(),
    )
    mock_load.return_value = model
    audio = tmp_path / "test.wav"
    audio.touch()

    with caplog.at_level("INFO"):
        transcribe(audio)

    assert "Transcribed 90 s" in caplog.text
