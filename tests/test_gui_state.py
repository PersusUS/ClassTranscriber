"""Unit tests for modules/gui_state.py — the logic behind the window.

No Tkinter involved: this is the part of the GUI that can be checked
without a display, and where the interesting decisions live.
"""

from datetime import datetime
from pathlib import Path

import pytest

import config
from modules import gui_state
from modules.gui_state import (
    STAGE_LABELS,
    FormState,
    build_options,
    default_session_name,
    describe_device,
    device_index_from_label,
    format_elapsed,
    initial_stages,
    meter_fraction,
    meter_verdict,
    missing_requirements,
    parse_duration_minutes,
    parse_speakers,
    sanitise_name,
)

VALID_TOKEN = "hf_" + "x" * 34


# ---------------------------------------------------------------------------
# Session naming
# ---------------------------------------------------------------------------

def test_default_session_name_sorts_chronologically():
    """The pre-filled name sorts by date, so sessions list in order."""
    name = default_session_name(datetime(2026, 3, 9, 8, 5))

    assert name == "clase_2026-03-09_08-05"


def test_sanitise_name_keeps_accents():
    """Spanish subject names must survive intact."""
    assert sanitise_name("Álgebra") == "Álgebra"


def test_sanitise_name_replaces_spaces():
    """Spaces become underscores so the filenames stay easy to handle."""
    assert sanitise_name("Cálculo II lunes") == "Cálculo_II_lunes"


def test_sanitise_name_strips_path_separators():
    """A name can never escape the output directory."""
    cleaned = sanitise_name("../../etc/passwd")

    assert "/" not in cleaned
    assert ".." not in cleaned


def test_sanitise_name_falls_back_when_empty():
    """An empty box still produces a usable session name."""
    assert sanitise_name("   ", now=datetime(2026, 3, 9, 8, 5)) == "clase_2026-03-09_08-05"


def test_sanitise_name_drops_dangerous_characters():
    """Quotes, wildcards and pipes are removed rather than escaped."""
    assert sanitise_name('cla"se*|?') == "clase"


# ---------------------------------------------------------------------------
# Form parsing
# ---------------------------------------------------------------------------

def test_parse_duration_empty_means_until_stopped():
    """An empty duration is the normal case: stop with the button."""
    assert parse_duration_minutes("") is None
    assert parse_duration_minutes("   ") is None


def test_parse_duration_reads_minutes():
    """The box is in minutes, which is how timetables are written."""
    assert parse_duration_minutes("90") == 90


def test_parse_duration_accepts_a_decimal_comma():
    """A Spanish keyboard produces "1,5" — accept it rather than rejecting."""
    assert parse_duration_minutes("1,5") == 1


def test_parse_duration_rejects_text():
    """Nonsense is reported in Spanish, before anything gets recorded."""
    with pytest.raises(ValueError, match="duración válida"):
        parse_duration_minutes("hora y media")


def test_parse_duration_rejects_zero_and_negative():
    """A zero-minute class is a typo, not an instruction."""
    with pytest.raises(ValueError, match="mayor que cero"):
        parse_duration_minutes("0")
    with pytest.raises(ValueError, match="mayor que cero"):
        parse_duration_minutes("-30")


def test_parse_speakers_empty_is_optional():
    """The speaker count is a hint, not a requirement."""
    assert parse_speakers("") is None


def test_parse_speakers_reads_a_count():
    assert parse_speakers("3") == 3


def test_parse_speakers_rejects_invalid():
    with pytest.raises(ValueError, match="número de personas"):
        parse_speakers("tres")
    with pytest.raises(ValueError, match="al menos una persona"):
        parse_speakers("0")


# ---------------------------------------------------------------------------
# Building the run
# ---------------------------------------------------------------------------

def test_build_options_translates_the_form():
    """Every field reaches the pipeline, with minutes converted to seconds."""
    form = FormState(
        name="Álgebra 1",
        device=2,
        profile="low",
        duration_minutes=90,
        num_speakers=3,
        denoise=False,
        diarize=True,
        clean=False,
    )

    options = build_options(form, hf_token=VALID_TOKEN)

    assert options.name == "Álgebra_1"
    assert options.device == 2
    assert options.duration == 5400
    assert options.num_speakers == 3
    assert options.denoise is False
    assert options.clean_enabled is False
    assert options.settings.profile.name == "low"
    assert options.hf_token == VALID_TOKEN


def test_build_options_is_never_interactive():
    """A window must not block on a terminal prompt nobody can see."""
    assert build_options(FormState(name="clase")).interactive is False


def test_build_options_identifies_the_professor_automatically():
    """With no terminal to ask in, the professor is the most talkative voice."""
    assert build_options(FormState(name="clase")).professor == "auto"


def test_build_options_no_duration_means_open_ended():
    """An empty duration records until the Stop button."""
    assert build_options(FormState(name="clase", duration_minutes=None)).duration is None


def test_build_options_passes_hooks_through():
    """The stop event and the callbacks reach the pipeline."""
    import threading

    event = threading.Event()
    options = build_options(
        FormState(name="clase"),
        stop_event=event,
        on_level=print,
        on_stage=print,
    )

    assert options.stop_event is event
    assert options.on_level is print
    assert options.on_stage is print


def test_build_options_defaults_to_spanish():
    """Classes are in Spanish, so the window does not need to ask."""
    assert build_options(FormState(name="clase")).settings.language == "es"


def test_build_options_carries_an_input_file():
    """Processing an existing recording skips the microphone entirely."""
    options = build_options(FormState(name="clase", input_path=Path("/tmp/clase.wav")))

    assert options.input_path == Path("/tmp/clase.wav")


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

def test_missing_requirements_flags_the_token():
    """Speaker separation without a token is caught before recording."""
    problems = missing_requirements(FormState(diarize=True, clean=False), None, ollama_ok=True)

    assert len(problems) == 1
    assert "HuggingFace" in problems[0]
    assert "Desmarca" in problems[0]        # Tells the user how to proceed anyway


def test_missing_requirements_flags_ollama():
    """LLM cleanup with the server down is caught before recording."""
    problems = missing_requirements(
        FormState(diarize=False, clean=True), VALID_TOKEN, ollama_ok=False
    )

    assert len(problems) == 1
    assert "ollama serve" in problems[0]


def test_missing_requirements_silent_when_features_are_off():
    """Turning both features off needs nothing installed."""
    form = FormState(diarize=False, clean=False)

    assert missing_requirements(form, None, ollama_ok=False) == []


def test_missing_requirements_reports_both():
    """Both problems are listed at once, not one per attempt."""
    problems = missing_requirements(FormState(), None, ollama_ok=False)

    assert len(problems) == 2


def test_missing_requirements_rejects_a_malformed_token():
    """A token that is not an hf_ token is treated as missing."""
    problems = missing_requirements(FormState(clean=False), "ghp_wrong_kind", ollama_ok=True)

    assert len(problems) == 1


# ---------------------------------------------------------------------------
# Level meter
# ---------------------------------------------------------------------------

def test_meter_fraction_spans_the_range():
    """The bar fills from the floor to full scale."""
    assert meter_fraction(gui_state.METER_FLOOR_DBFS) == 0.0
    assert meter_fraction(0.0) == 1.0
    assert meter_fraction(-30.0) == pytest.approx(0.5)


def test_meter_fraction_clamps():
    """Levels outside the range do not overflow the bar."""
    assert meter_fraction(-120.0) == 0.0
    assert meter_fraction(6.0) == 1.0


def test_meter_verdict_warns_when_quiet():
    """The most useful message in the window: the mic is too far away."""
    state, message = meter_verdict(-55.0)

    assert state == "quiet"
    assert "acerca el micrófono" in message.lower()


def test_meter_verdict_warns_when_clipping():
    state, message = meter_verdict(-0.2)

    assert state == "clipping"
    assert "satura" in message


def test_meter_verdict_ok_in_between():
    assert meter_verdict(-18.0) == ("ok", "Nivel correcto")


def test_meter_verdict_matches_the_recorder_thresholds():
    """The window and the log must not disagree about what is too quiet."""
    assert gui_state.QUIET_PEAK_DBFS == -45.0
    assert gui_state.CLIPPING_PEAK_DBFS == -1.0


def test_format_elapsed():
    """Elapsed time reads as MM:SS, growing to H:MM:SS for a long class."""
    assert format_elapsed(0) == "00:00"
    assert format_elapsed(95) == "01:35"
    assert format_elapsed(3725) == "1:02:05"


# ---------------------------------------------------------------------------
# Stage checklist
# ---------------------------------------------------------------------------

def test_initial_stages_cover_every_pipeline_stage():
    """The checklist and the pipeline's stage list cannot drift apart."""
    assert set(initial_stages(FormState())) == set(pipeline_stage_keys())


def pipeline_stage_keys():
    from modules.pipeline import STAGES

    return STAGES


def test_initial_stages_premarks_skipped_work():
    """What this run will not do is greyed out from the start."""
    form = FormState(diarize=False, clean=False, input_path=Path("/tmp/a.wav"))

    stages = initial_stages(form)

    assert stages["record"] == "skipped"
    assert stages["diarize"] == "skipped"
    assert stages["clean"] == "skipped"
    assert stages["transcribe"] == "pending"


def test_stage_labels_are_in_spanish():
    """The window is in Spanish, like the classes."""
    labels = dict(STAGE_LABELS)

    assert labels["record"] == "Grabar"
    assert labels["transcribe"] == "Transcribir"


# ---------------------------------------------------------------------------
# Device dropdown
# ---------------------------------------------------------------------------

def test_describe_device_marks_the_default():
    label = describe_device({"index": 1, "name": "Micro USB", "default": True})

    assert label == "1: Micro USB (predeterminado)"


def test_device_index_roundtrip():
    """The index survives the trip through the dropdown label."""
    device = {"index": 3, "name": "Micro: raro", "default": False}

    assert device_index_from_label(describe_device(device)) == 3


def test_device_index_of_placeholder_is_none():
    """The "no microphone" placeholder does not parse as a device."""
    assert device_index_from_label("sin micrófono detectado") is None
    assert device_index_from_label("") is None
