"""Unit tests for modules/pipeline.py — orchestration and resuming."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config
from modules import pipeline
from modules.pipeline import PipelineOptions, run_pipeline

TRANSCRIPTION = [
    {"start": 0.0, "end": 4.0, "text": "Buenos días, empezamos el tema tres."},
    {"start": 4.5, "end": 6.0, "text": "¿Puede repetir?"},
    {"start": 6.5, "end": 12.0, "text": "Claro, el tema tres trata de derivadas."},
]

DIARIZATION = [
    {"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00"},
    {"start": 4.3, "end": 6.1, "speaker": "SPEAKER_01"},
    {"start": 6.4, "end": 12.0, "speaker": "SPEAKER_00"},
]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Redirects every output directory into a temporary workspace."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path / "audio")
    return tmp_path


@pytest.fixture
def recording(tmp_path):
    """A stand-in audio file; every stage that reads it is mocked."""
    path = tmp_path / "clase.wav"
    path.write_bytes(b"RIFF")
    return path


def _options(recording: Path, **overrides) -> PipelineOptions:
    defaults = dict(
        name="clase01",
        settings=config.resolve_settings(profile="low", device="cpu"),
        input_path=recording,
        interactive=False,
        professor="auto",
        formats=("txt", "professor", "json"),
    )
    defaults.update(overrides)
    return PipelineOptions(**defaults)


@pytest.fixture
def stages():
    """Patches every heavyweight stage with a fast fake."""
    with patch("modules.pipeline.preprocess") as preprocess, \
         patch("modules.pipeline.transcribe") as transcribe, \
         patch("modules.pipeline.diarize") as diarize, \
         patch("modules.pipeline.clean_transcript") as clean, \
         patch("modules.pipeline.record") as record:

        preprocess.side_effect = lambda source, target, **kwargs: Path(target).write_bytes(b"RIFF")
        transcribe.return_value = TRANSCRIPTION
        diarize.return_value = DIARIZATION
        clean.side_effect = lambda segments, **kwargs: segments

        yield {
            "preprocess": preprocess,
            "transcribe": transcribe,
            "diarize": diarize,
            "clean": clean,
            "record": record,
        }


def test_pipeline_writes_professor_transcript(workspace, recording, stages):
    """The headline deliverable: a file with only the professor's words."""
    outputs = run_pipeline(_options(recording))

    professor_text = outputs["professor"].read_text(encoding="utf-8")

    assert "derivadas" in professor_text
    assert "¿Puede repetir?" not in professor_text


def test_pipeline_writes_full_transcript(workspace, recording, stages):
    """The full transcript keeps every speaker, professor renamed."""
    outputs = run_pipeline(_options(recording))

    full_text = outputs["full"].read_text(encoding="utf-8")

    assert "PROFESOR:" in full_text
    assert "¿Puede repetir?" in full_text


def test_pipeline_does_not_record_when_given_a_file(workspace, recording, stages):
    """Passing --input must skip the recording stage entirely."""
    run_pipeline(_options(recording))

    stages["record"].assert_not_called()


def test_pipeline_caches_every_stage(workspace, recording, stages):
    """Intermediate results land in the session directory for reuse."""
    run_pipeline(_options(recording))

    session = config.SESSIONS_DIR / "clase01"
    for artifact in ("transcription.json", "diarization.json", "merged.json", "cleaned.json"):
        assert (session / artifact).exists()


def test_pipeline_resume_skips_completed_stages(workspace, recording, stages):
    """--resume reuses cached work instead of re-running the slow stages."""
    run_pipeline(_options(recording))
    stages["transcribe"].reset_mock()
    stages["diarize"].reset_mock()

    run_pipeline(_options(recording, resume=True))

    stages["transcribe"].assert_not_called()
    stages["diarize"].assert_not_called()


def test_pipeline_reruns_stages_without_resume(workspace, recording, stages):
    """Without --resume everything is recomputed."""
    run_pipeline(_options(recording))
    stages["transcribe"].reset_mock()

    run_pipeline(_options(recording))

    stages["transcribe"].assert_called_once()


def test_pipeline_without_diarization(workspace, recording, stages):
    """--no-diarize still produces a transcript, with no HuggingFace token."""
    outputs = run_pipeline(_options(recording, diarize_enabled=False))

    stages["diarize"].assert_not_called()
    assert "derivadas" in outputs["professor"].read_text(encoding="utf-8")


def test_pipeline_without_cleanup(workspace, recording, stages):
    """--no-clean skips Ollama entirely."""
    run_pipeline(_options(recording, clean_enabled=False))

    stages["clean"].assert_not_called()
    assert not (config.SESSIONS_DIR / "clase01" / "cleaned.json").exists()


def test_pipeline_cleans_only_the_professor(workspace, recording, stages):
    """The LLM is pointed at the professor, not at the whole room."""
    run_pipeline(_options(recording))

    assert stages["clean"].call_args.kwargs["professor"] == "SPEAKER_00"


def test_pipeline_missing_input(workspace, tmp_path, stages):
    """A bad --input path fails immediately with a clear error."""
    options = _options(tmp_path / "no_existe.wav")

    with pytest.raises(FileNotFoundError):
        run_pipeline(options)


def test_pipeline_rejects_empty_transcription(workspace, recording, stages):
    """Silence in, actionable error out — not an empty file."""
    stages["transcribe"].return_value = []

    with pytest.raises(RuntimeError, match="no usable speech"):
        run_pipeline(_options(recording))


def test_pipeline_writes_speaker_stats(workspace, recording, stages):
    """The speaker breakdown is saved so the choice can be reviewed."""
    outputs = run_pipeline(_options(recording))

    stats = json.loads(outputs["speakers"].read_text(encoding="utf-8"))

    assert stats[0]["speaker"] == "SPEAKER_00"


def test_clear_session_removes_cache(workspace, recording, stages):
    """`clear` deletes the cached intermediates for a session."""
    run_pipeline(_options(recording))
    assert (config.SESSIONS_DIR / "clase01").exists()

    pipeline.clear_session("clase01")

    assert not (config.SESSIONS_DIR / "clase01").exists()
