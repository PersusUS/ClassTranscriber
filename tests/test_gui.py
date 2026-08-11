"""Unit tests for gui.py — the Tkinter window.

These need Tkinter and a display, so they skip automatically where either
is missing (a headless server, or a Python built without Tk). On Windows
and macOS, and on Linux under Xvfb, they run.

Nothing here touches audio hardware or a model: the pipeline is patched
out, and what is checked is that the window wires the user's choices to
the run and reflects the run's progress back.
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

tkinter = pytest.importorskip("tkinter", reason="Tkinter is not installed")

FAKE_DEVICES = [
    {"index": 0, "name": "Micrófono interno", "channels": 2, "default": True},
    {"index": 1, "name": "Micrófono USB", "channels": 1, "default": False},
]

VALID_TOKEN = "hf_" + "x" * 34


@pytest.fixture
def root():
    """A Tk root, or a skip when there is no display to draw on."""
    try:
        window = tkinter.Tk()
    except tkinter.TclError as exc:      # pragma: no cover - depends on the host
        pytest.skip(f"No display available: {exc}")
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def app(root, monkeypatch):
    """The window, with the microphone list and Ollama check stubbed."""
    import gui

    monkeypatch.setattr("main.list_input_devices", lambda: list(FAKE_DEVICES))
    monkeypatch.setattr("main.check_ollama", lambda: (True, "ok"))
    monkeypatch.setenv("HF_TOKEN", VALID_TOKEN)

    instance = gui.build_app(root)
    root.update()
    return instance


# ---------------------------------------------------------------------------
# Opening the window
# ---------------------------------------------------------------------------

def test_window_prefills_a_session_name(app):
    """There is nothing to type to get started."""
    assert app.name_var.get().startswith("clase_")


def test_window_lists_the_microphones(app):
    """Both devices appear, the default one selected."""
    assert len(app.device_box["values"]) == 2
    assert app.device_var.get().startswith("0: Micrófono interno")


def test_window_survives_having_no_microphone(root, monkeypatch):
    """A machine with no input device still opens, with a clear placeholder."""
    import gui

    monkeypatch.setattr("main.list_input_devices", lambda: [])
    instance = gui.build_app(root)
    root.update()

    assert instance.device_var.get() == "sin micrófono detectado"


def test_window_survives_a_portaudio_failure(root, monkeypatch):
    """A PortAudio error is logged in the window instead of crashing it."""
    import gui

    monkeypatch.setattr("main.list_input_devices", MagicMock(side_effect=OSError("no PortAudio")))
    instance = gui.build_app(root)
    root.update()

    assert instance.devices == []
    assert "no se pudo leer la lista" in instance.log.get("1.0", "end").lower()


def test_stop_button_starts_disabled(app):
    """Nothing to stop before anything is recording."""
    assert str(app.stop_button.cget("state")) == "disabled"


def test_open_button_starts_disabled(app):
    """No transcript to open yet."""
    assert str(app.open_button.cget("state")) == "disabled"


# ---------------------------------------------------------------------------
# Reading the form
# ---------------------------------------------------------------------------

def test_read_form_reflects_the_widgets(app):
    """What the user picks is what the form reports."""
    app.name_var.set("Álgebra")
    app.duration_var.set("90")
    app.speakers_var.set("3")
    app.profile_var.set("low")
    app.denoise_var.set(False)

    form = app.read_form()

    assert form.name == "Álgebra"
    assert form.duration_minutes == 90
    assert form.num_speakers == 3
    assert form.profile == "low"
    assert form.denoise is False
    assert form.device == 0


def test_read_form_rejects_a_bad_duration(app):
    """An unparseable duration raises rather than silently recording forever."""
    app.duration_var.set("hora y media")

    with pytest.raises(ValueError):
        app.read_form()


# ---------------------------------------------------------------------------
# Starting a run
# ---------------------------------------------------------------------------

def test_start_reports_a_bad_duration_and_does_not_run(app):
    """A typo produces a dialog, not a two-hour recording."""
    app.duration_var.set("noventa")

    with patch("gui.messagebox.showerror") as error, \
         patch("gui.pipeline.run_pipeline") as run:
        app.start_recording()

    error.assert_called_once()
    run.assert_not_called()


def test_start_blocks_when_ollama_is_down(app, monkeypatch):
    """Discovering Ollama is down after the class would be too late."""
    monkeypatch.setattr("main.check_ollama", lambda: (False, "refused"))

    with patch("gui.messagebox.showerror") as error, \
         patch("gui.pipeline.run_pipeline") as run:
        app.start_recording()

    assert "ollama serve" in error.call_args.args[1]
    run.assert_not_called()


def test_start_blocks_when_the_token_is_missing(app, monkeypatch):
    """Same for speaker separation without a HuggingFace token."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with patch("gui.messagebox.showerror") as error, \
         patch("gui.pipeline.run_pipeline") as run:
        app.start_recording()

    assert "HuggingFace" in error.call_args.args[1]
    run.assert_not_called()


def test_start_proceeds_with_features_turned_off(app, monkeypatch):
    """Unticking both boxes lets a bare machine run.

    This is the escape hatch the error dialogs point at, so it has to work.
    """
    monkeypatch.setattr("main.check_ollama", lambda: (False, "refused"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    app.diarize_var.set(False)
    app.clean_var.set(False)

    with patch("gui.pipeline.run_pipeline", return_value={}) as run:
        app.start_recording()
        app.worker.join(timeout=5)

    run.assert_called_once()


def test_start_passes_the_form_to_the_pipeline(app):
    """The options the pipeline receives match what the window showed."""
    app.name_var.set("Cálculo")
    app.duration_var.set("45")
    app.speakers_var.set("2")

    with patch("gui.pipeline.run_pipeline", return_value={}) as run:
        app.start_recording()
        app.worker.join(timeout=5)

    options = run.call_args.args[0]
    assert options.name == "Cálculo"
    assert options.duration == 2700
    assert options.num_speakers == 2
    assert options.interactive is False
    assert options.stop_event is app.stop_event


def test_process_file_skips_recording(app, tmp_path):
    """Choosing a file runs the pipeline on it, with no microphone involved."""
    recording = tmp_path / "clase.wav"
    recording.write_bytes(b"RIFF")

    with patch("gui.filedialog.askopenfilename", return_value=str(recording)), \
         patch("gui.pipeline.run_pipeline", return_value={}) as run:
        app.process_file()
        app.worker.join(timeout=5)

    assert run.call_args.args[0].input_path == recording


def test_process_file_cancelled_does_nothing(app):
    """Closing the file dialog is not a request to record."""
    with patch("gui.filedialog.askopenfilename", return_value=""), \
         patch("gui.pipeline.run_pipeline") as run:
        app.process_file()

    run.assert_not_called()


def test_start_is_ignored_while_a_run_is_active(app):
    """Double-clicking Record must not launch two pipelines."""
    release = threading.Event()

    def blocking(_options):
        release.wait(timeout=5)
        return {}

    with patch("gui.pipeline.run_pipeline", side_effect=blocking) as run:
        app.start_recording()
        app.start_recording()
        release.set()
        app.worker.join(timeout=5)

    assert run.call_count == 1


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------

def test_stop_sets_the_event_the_recorder_watches(app):
    """The Stop button is how a GUI ends an open-ended recording."""
    with patch("gui.pipeline.run_pipeline", return_value={}):
        app.start_recording()
        app.worker.join(timeout=5)

    app.stop_recording()

    assert app.stop_event.is_set()


# ---------------------------------------------------------------------------
# Progress coming back from the worker
# ---------------------------------------------------------------------------

def test_level_updates_the_meter(app):
    """A healthy level fills the bar and says so."""
    app._apply_level(37.0, -22.0, -14.0)

    assert 0.5 < app.meter.cget("value") < 0.8
    assert "00:37" in app.meter_text.get()
    assert "correcto" in app.meter_text.get()


def test_level_warns_when_the_microphone_is_too_far(app):
    """The window's most useful message during a class."""
    app._apply_level(10.0, -58.0, -52.0)

    assert "acerca el micrófono" in app.meter_text.get().lower()


def test_stage_marks_progress(app):
    """Finished stages get a tick, the current one an arrow."""
    app._apply_stage("preprocess", "done")
    app._apply_stage("transcribe", "running")

    assert app.stage_labels["preprocess"].cget("text").startswith("✓")
    assert app.stage_labels["transcribe"].cget("text").startswith("▶")


def test_recording_finished_re_enables_nothing_yet(app):
    """When recording ends the Stop button goes away, but work continues."""
    app._apply_stage("record", "done")

    assert str(app.stop_button.cget("state")) == "disabled"
    assert str(app.record_button.cget("state")) == "disabled"


def test_done_enables_opening_the_transcript(app, tmp_path):
    """Finishing offers the professor transcript straight away."""
    professor = tmp_path / "clase_profesor.txt"
    professor.write_text("texto", encoding="utf-8")

    app._apply_done({"professor": professor})

    assert str(app.open_button.cget("state")) == "normal"
    assert str(app.record_button.cget("state")) == "normal"
    assert "clase_profesor.txt" in app.log.get("1.0", "end")


def test_failure_is_shown_to_the_user(app):
    """A failed run says why, in a dialog and in the log."""
    with patch("gui.messagebox.showerror") as error:
        app._apply_failure(RuntimeError("Ollama no responde"))

    error.assert_called_once()
    assert "Ollama no responde" in app.log.get("1.0", "end")
    assert str(app.record_button.cget("state")) == "normal"


def test_failure_in_the_worker_reaches_the_window(app):
    """An exception in the worker thread is reported, not swallowed."""
    with patch("gui.pipeline.run_pipeline", side_effect=RuntimeError("sin voz utilizable")):
        app.start_recording()
        app.worker.join(timeout=5)

    kind, payload = app.messages.get(timeout=5)

    assert kind == "failed"
    assert "sin voz utilizable" in str(payload)


def test_drain_applies_queued_messages(app):
    """The queue is what carries worker updates into the UI thread."""
    app.messages.put(("stage", ("export", "done")))
    app.messages.put(("level", (5.0, -20.0, -12.0)))

    app._drain()

    assert app.stage_labels["export"].cget("text").startswith("✓")
    assert "00:05" in app.meter_text.get()


def test_open_result_without_output_does_nothing(app):
    """The button cannot open a transcript that was never written."""
    app.outputs = {}
    app.open_result()      # Must not raise


# ---------------------------------------------------------------------------
# Opening the finished transcript
# ---------------------------------------------------------------------------

def test_open_result_uses_the_platform_opener(app, tmp_path, monkeypatch):
    """On Linux the transcript is handed to the desktop's default handler."""
    professor = tmp_path / "clase_profesor.txt"
    professor.write_text("texto", encoding="utf-8")
    app.outputs = {"professor": professor}

    monkeypatch.setattr("gui.sys.platform", "linux")
    with patch("gui.webbrowser.open") as opener:
        app.open_result()

    assert opener.call_args.args[0] == professor.as_uri()


def test_open_result_uses_open_on_macos(app, tmp_path, monkeypatch):
    """macOS gets `open`, which respects the user's chosen editor."""
    professor = tmp_path / "clase_profesor.txt"
    professor.write_text("texto", encoding="utf-8")
    app.outputs = {"professor": professor}

    monkeypatch.setattr("gui.sys.platform", "darwin")
    with patch("gui.subprocess.run") as runner:
        app.open_result()

    assert runner.call_args.args[0] == ["open", str(professor)]


def test_open_result_falls_back_to_the_full_transcript(app, tmp_path, monkeypatch):
    """With professor filtering off, the full transcript is offered instead."""
    full = tmp_path / "clase_completo.txt"
    full.write_text("texto", encoding="utf-8")
    app.outputs = {"full": full}

    monkeypatch.setattr("gui.sys.platform", "linux")
    with patch("gui.webbrowser.open") as opener:
        app.open_result()

    assert opener.call_args.args[0] == full.as_uri()


def test_open_result_reports_the_path_if_it_cannot_open(app, tmp_path, monkeypatch):
    """If nothing can open the file, the window at least says where it is."""
    professor = tmp_path / "clase_profesor.txt"
    professor.write_text("texto", encoding="utf-8")
    app.outputs = {"professor": professor}

    monkeypatch.setattr("gui.sys.platform", "linux")
    with patch("gui.webbrowser.open", side_effect=OSError("no handler")), \
         patch("gui.messagebox.showinfo") as info:
        app.open_result()

    assert str(professor) in info.call_args.args[1]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_start_without_a_microphone_is_refused(root, monkeypatch):
    """Recording with no input device explains the alternative."""
    import gui

    monkeypatch.setattr("main.list_input_devices", lambda: [])
    monkeypatch.setattr("main.check_ollama", lambda: (True, "ok"))
    monkeypatch.setenv("HF_TOKEN", VALID_TOKEN)
    instance = gui.build_app(root)

    with patch("gui.messagebox.showerror") as error, \
         patch("gui.pipeline.run_pipeline") as run:
        instance.start_recording()

    assert "Procesar un audio" in error.call_args.args[1]
    run.assert_not_called()


def test_callbacks_only_queue_messages(app):
    """The worker's callbacks must not touch widgets — Tkinter is not thread-safe."""
    app._on_level(1.0, -20.0, -10.0)
    app._on_stage("merge", "running")

    assert app.messages.qsize() == 2
    assert app.messages.get_nowait()[0] == "level"
    assert app.messages.get_nowait()[0] == "stage"


def test_drain_is_safe_when_idle(app):
    """Draining an empty queue is a no-op, not an exception."""
    app._drain()


def test_run_opens_and_closes_a_window(monkeypatch, tmp_path):
    """`gui.run()` builds a real window and enters the loop."""
    import gui

    monkeypatch.setattr("config.AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr("config.OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr("config.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("main.list_input_devices", lambda: list(FAKE_DEVICES))

    with patch("gui.Tk") as tk_class:
        root = MagicMock()
        tk_class.return_value = root
        with patch("gui.build_app") as build:
            gui.run()

    build.assert_called_once_with(root)
    root.mainloop.assert_called_once()
    assert (tmp_path / "output").is_dir()


def test_drain_delivers_the_finished_run(app, tmp_path):
    """In production the result reaches the widgets through the queue.

    Calling the handler directly skips the dispatch, which is where a
    wrong message key would hide.
    """
    professor = tmp_path / "clase_profesor.txt"
    professor.write_text("texto", encoding="utf-8")
    app.messages.put(("done", {"professor": professor}))

    app._drain()

    assert str(app.open_button.cget("state")) == "normal"
    assert app.outputs["professor"] == professor


def test_drain_delivers_a_failure(app):
    """A failed run surfaces through the same queue."""
    app.messages.put(("failed", RuntimeError("sin voz utilizable")))

    with patch("gui.messagebox.showerror") as error:
        app._drain()

    error.assert_called_once()
    assert "sin voz utilizable" in app.log.get("1.0", "end")


def test_full_round_trip_through_the_queue(app, tmp_path):
    """The worker runs, reports progress and its result lands in the window."""
    professor = tmp_path / "clase_profesor.txt"
    professor.write_text("texto", encoding="utf-8")

    def fake_pipeline(options):
        options.on_stage("record", "running")
        options.on_level(12.0, -21.0, -13.0)
        options.on_stage("record", "done")
        options.on_stage("export", "done")
        return {"professor": professor}

    with patch("gui.pipeline.run_pipeline", side_effect=fake_pipeline):
        app.start_recording()
        app.worker.join(timeout=5)

    app._drain()

    assert app.stage_labels["export"].cget("text").startswith("✓")
    assert "00:12" in app.meter_text.get() or "Listo" in app.meter_text.get()
    assert str(app.open_button.cget("state")) == "normal"
    assert str(app.record_button.cget("state")) == "normal"
