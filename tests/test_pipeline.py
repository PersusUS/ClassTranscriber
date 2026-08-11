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


def _options(recording: Path | None, **overrides) -> PipelineOptions:
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


def test_pipeline_records_when_no_input_given(workspace, stages):
    """Without --input the pipeline records first."""
    stages["record"].side_effect = lambda path, duration, **kw: Path(path).write_bytes(b"RIFF")

    run_pipeline(_options(None, input_path=None, duration=60))

    stages["record"].assert_called_once()
    assert stages["record"].call_args.args[1] == 60


def test_pipeline_reuses_an_existing_recording_on_resume(workspace, stages):
    """--resume does not re-record over a session's existing audio."""
    session = config.SESSIONS_DIR / "clase01"
    session.mkdir(parents=True)
    (session / "raw.wav").write_bytes(b"RIFF")

    run_pipeline(_options(None, input_path=None, resume=True))

    stages["record"].assert_not_called()


def test_pipeline_never_deletes_the_users_own_file(workspace, recording, stages):
    """--no-keep-audio must not delete a recording the user supplied."""
    run_pipeline(_options(recording, keep_audio=False))

    assert recording.exists()


def test_pipeline_deletes_its_own_recording_when_asked(workspace, stages):
    """--no-keep-audio does delete audio the pipeline recorded itself."""
    stages["record"].side_effect = lambda path, duration, **kw: Path(path).write_bytes(b"RIFF")

    run_pipeline(_options(None, input_path=None, keep_audio=False))

    assert not (config.SESSIONS_DIR / "clase01" / "raw.wav").exists()


def test_pipeline_passes_speaker_hints_to_diarization(workspace, recording, stages):
    """--num-speakers reaches pyannote, where it improves accuracy."""
    run_pipeline(_options(recording, num_speakers=3))

    assert stages["diarize"].call_args.kwargs["num_speakers"] == 3


def test_pipeline_forwards_denoise_flag(workspace, recording, stages):
    """--no-denoise reaches the preprocessor."""
    run_pipeline(_options(recording, denoise=False))

    assert stages["preprocess"].call_args.kwargs["denoise"] is False


def test_pipeline_respects_explicit_professor(workspace, recording, stages):
    """An explicit --professor overrides the airtime heuristic."""
    run_pipeline(_options(recording, professor="SPEAKER_01"))

    assert stages["clean"].call_args.kwargs["professor"] == "SPEAKER_01"


def test_pipeline_professor_none_skips_the_professor_file(workspace, recording, stages):
    """--professor none produces the full transcript only."""
    outputs = run_pipeline(_options(recording, professor="none"))

    assert "professor" not in outputs
    assert "full" in outputs


def test_pipeline_cleans_a_copy_not_the_merged_cache(workspace, recording, stages):
    """The merged cache must stay pristine so --resume can re-clean it.

    Cleaning in place would make a second run clean already-cleaned text.
    """
    stages["clean"].side_effect = lambda segments, **kw: [
        {**segment, "text": "REWRITTEN"} for segment in segments
    ]

    run_pipeline(_options(recording))

    merged = json.loads((config.SESSIONS_DIR / "clase01" / "merged.json").read_text())
    assert all(segment["text"] != "REWRITTEN" for segment in merged)


def test_pipeline_uses_word_level_attribution(workspace, recording, stages):
    """Speaker labels come from the word timings when they are available."""
    stages["transcribe"].return_value = [
        {
            "start": 0.0, "end": 6.0, "text": "Explico y luego preguntan",
            "words": [
                {"start": 0.0, "end": 1.0, "word": " Explico"},
                {"start": 1.0, "end": 2.0, "word": " y"},
                {"start": 2.0, "end": 3.0, "word": " luego"},
                {"start": 4.6, "end": 5.4, "word": " preguntan"},
            ],
        }
    ]
    stages["diarize"].return_value = [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
        {"start": 4.3, "end": 6.0, "speaker": "SPEAKER_01"},
    ]

    outputs = run_pipeline(_options(recording))
    professor_text = outputs["professor"].read_text(encoding="utf-8")

    assert "Explico y luego" in professor_text
    assert "preguntan" not in professor_text


def test_pipeline_logs_the_resolved_settings(workspace, recording, stages, caplog):
    """The run starts by stating which model and device it will use."""
    with caplog.at_level("INFO"):
        run_pipeline(_options(recording))

    assert "profile=low" in caplog.text
    assert "device=cpu" in caplog.text


def test_pipeline_reports_stage_progress(workspace, recording, stages):
    """The GUI's checklist is driven by these callbacks."""
    seen = []

    run_pipeline(_options(recording, on_stage=lambda key, status: seen.append((key, status))))

    assert ("transcribe", "running") in seen
    assert ("transcribe", "done") in seen
    assert ("export", "done") in seen
    # A supplied recording means nothing was recorded.
    assert ("record", "skipped") in seen


def test_pipeline_marks_disabled_stages_as_skipped(workspace, recording, stages):
    """Turning a feature off shows as skipped, not as pending forever."""
    seen = []

    run_pipeline(
        _options(
            recording,
            diarize_enabled=False,
            clean_enabled=False,
            on_stage=lambda key, status: seen.append((key, status)),
        )
    )

    assert ("diarize", "skipped") in seen
    assert ("clean", "skipped") in seen


def test_pipeline_reports_reused_stages_as_skipped(workspace, recording, stages):
    """On --resume the cached stages are marked skipped for the user."""
    run_pipeline(_options(recording))
    seen = []

    run_pipeline(_options(recording, resume=True,
                          on_stage=lambda key, status: seen.append((key, status))))

    assert ("transcribe", "skipped") in seen
    assert ("diarize", "skipped") in seen


def test_pipeline_forwards_the_stop_event_and_meter(workspace, stages):
    """Both GUI hooks reach the recorder."""
    import threading

    event = threading.Event()
    meter = lambda *args: None      # noqa: E731 - identity is what is checked
    stages["record"].side_effect = lambda path, duration, **kw: Path(path).write_bytes(b"RIFF")

    run_pipeline(_options(None, input_path=None, stop_event=event, on_level=meter))

    assert stages["record"].call_args.kwargs["stop_event"] is event
    assert stages["record"].call_args.kwargs["on_level"] is meter


def test_pipeline_survives_a_broken_progress_callback(workspace, recording, stages):
    """A crashing GUI callback must not take the transcript down with it."""
    def explode(key, status):
        raise ValueError("ventana cerrada")

    outputs = run_pipeline(_options(recording, on_stage=explode))

    assert outputs["professor"].exists()


def test_stage_list_matches_the_labels():
    """The pipeline's stages and the window's labels stay in step."""
    from modules.gui_state import STAGE_LABELS
    from modules.pipeline import STAGES

    assert tuple(key for key, _ in STAGE_LABELS) == STAGES
