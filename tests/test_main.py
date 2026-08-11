"""Unit tests for main.py — the CLI surface.

These cover argument parsing, the per-command prerequisite logic and the
dispatch of each subcommand. Every heavyweight stage is mocked: the point
is that the CLI wires the right arguments to the right function, not that
Whisper works.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config
import main

MERGED = [
    {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00", "text": "Explicación del profesor."},
    {"start": 10.0, "end": 12.0, "speaker": "SPEAKER_01", "text": "¿Una pregunta?"},
]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Redirects the project directories into a temporary workspace."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path / "audio")
    return tmp_path


@pytest.fixture
def merged_json(tmp_path) -> Path:
    """A merged-transcript JSON file on disk."""
    path = tmp_path / "merged.json"
    path.write_text(json.dumps(MERGED), encoding="utf-8")
    return path


def _parse(*argv):
    """Parses a CLI invocation without running it."""
    return main.build_parser().parse_args(list(argv))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_parser_requires_a_subcommand():
    """Running with no command is an error, not a silent no-op."""
    with pytest.raises(SystemExit):
        main.build_parser().parse_args([])


def test_parser_run_defaults():
    """`run` defaults to Spanish, auto profile and a two-hour cap."""
    args = _parse("run", "--name", "clase")

    assert args.name == "clase"
    assert args.duration == config.DEFAULT_DURATION
    assert args.language == "es"
    assert args.input is None
    assert args.no_diarize is False


def test_parser_run_requires_name():
    """A session name is mandatory — it names every output file."""
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["run"])


def test_parser_global_flags_before_subcommand():
    """Global flags are accepted ahead of the subcommand."""
    args = _parse("--profile", "low", "--device", "cpu", "--threads", "2", "run", "--name", "c")

    assert (args.profile, args.device, args.threads) == ("low", "cpu", 2)


def test_parser_record_duration_optional():
    """`record` with no duration records until interrupted."""
    assert _parse("record", "--output", "a.wav").duration is None


def test_parser_negative_flags():
    """The opt-out flags parse as booleans."""
    args = _parse("run", "--name", "c", "--no-diarize", "--no-clean", "--no-denoise", "--resume")

    assert (args.no_diarize, args.no_clean, args.no_denoise, args.resume) == (
        True, True, True, True,
    )


# ---------------------------------------------------------------------------
# Per-command prerequisites
# ---------------------------------------------------------------------------

def test_requirements_record_needs_only_a_microphone():
    """Recording must not demand Ollama or a HuggingFace token."""
    assert main.requirements_for(_parse("record", "--output", "a.wav")) == {"microphone"}


def test_requirements_preprocess_needs_nothing():
    """Preprocessing is pure DSP."""
    assert main.requirements_for(_parse("preprocess", "--input", "a", "--output", "b")) == set()


def test_requirements_transcribe_needs_nothing():
    """Transcription needs no token and no server."""
    assert main.requirements_for(_parse("transcribe", "--input", "a")) == set()


def test_requirements_diarize_needs_token():
    """Only diarization needs the gated pyannote model."""
    assert main.requirements_for(_parse("diarize", "--input", "a")) == {"hf_token"}


def test_requirements_clean_needs_ollama():
    """Only the cleanup stage needs the LLM server."""
    assert main.requirements_for(_parse("clean", "--input", "a")) == {"ollama"}


def test_requirements_run_full():
    """A full run from the microphone needs all three."""
    assert main.requirements_for(_parse("run", "--name", "c")) == {
        "microphone", "hf_token", "ollama",
    }


def test_requirements_run_with_input_skips_microphone():
    """Processing an existing file needs no microphone."""
    requires = main.requirements_for(_parse("run", "--name", "c", "--input", "a.wav"))

    assert "microphone" not in requires


def test_requirements_run_opt_outs():
    """--no-diarize and --no-clean drop their prerequisites."""
    requires = main.requirements_for(
        _parse("run", "--name", "c", "--input", "a.wav", "--no-diarize", "--no-clean")
    )

    assert requires == set()


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

def test_check_ollama_reachable():
    """A successful HTTP call reports the server as reachable."""
    with patch("main.urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = MagicMock()
        ok, detail = main.check_ollama()

    assert ok is True
    assert config.OLLAMA_HOST in detail


def test_check_ollama_unreachable():
    """A refused connection is reported, not raised."""
    with patch("main.urllib.request.urlopen", side_effect=OSError("refused")):
        ok, detail = main.check_ollama()

    assert ok is False
    assert "not reachable" in detail


def test_check_hf_token_missing(monkeypatch):
    """A missing token is reported with what it is needed for."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ok, detail = main.check_hf_token()

    assert ok is False
    assert "diarization" in detail


def test_check_hf_token_malformed(monkeypatch):
    """A token that is not an hf_ token is caught before any download."""
    monkeypatch.setenv("HF_TOKEN", "ghp_not_a_huggingface_token")

    assert main.check_hf_token()[0] is False


def test_check_hf_token_valid(monkeypatch):
    """A well-formed token passes."""
    monkeypatch.setenv("HF_TOKEN", "hf_" + "x" * 34)

    assert main.check_hf_token() == (True, "configured")


def test_check_microphone_reports_default():
    """The default device is named in the report."""
    with patch("main.list_input_devices") as devices:
        devices.return_value = [
            {"index": 0, "name": "Mic interno", "channels": 1, "default": True},
            {"index": 1, "name": "USB", "channels": 2, "default": False},
        ]
        ok, detail = main.check_microphone()

    assert ok is True
    assert "Mic interno" in detail


def test_check_microphone_no_devices():
    """No input device is a MISSING result, not a crash."""
    with patch("main.list_input_devices", return_value=[]):
        assert main.check_microphone()[0] is False


def test_check_microphone_portaudio_missing():
    """A PortAudio import or driver failure is reported gracefully."""
    with patch("main.list_input_devices", side_effect=OSError("no PortAudio")):
        ok, detail = main.check_microphone()

    assert ok is False
    assert "could not query" in detail


def test_validate_environment_creates_directories(workspace):
    """The output directories are created on demand."""
    main.validate_environment(set())

    assert config.OUTPUT_DIR.is_dir()
    assert config.SESSIONS_DIR.is_dir()
    assert config.AUDIO_DIR.is_dir()


def test_validate_environment_exits_on_missing_requirement(workspace, monkeypatch):
    """A missing prerequisite stops the command before it does any work."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        main.validate_environment({"hf_token"})

    assert exit_info.value.code == 1


def test_validate_environment_passes_when_satisfied(workspace, monkeypatch):
    """A satisfied requirement does not exit."""
    monkeypatch.setenv("HF_TOKEN", "hf_" + "x" * 34)

    main.validate_environment({"hf_token"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_settings_from_args_honours_flags():
    """Global flags reach the resolved settings."""
    settings = main.settings_from_args(
        _parse("--profile", "low", "--device", "cpu", "--threads", "2", "doctor")
    )

    assert (settings.profile.name, settings.device, settings.cpu_threads) == ("low", "cpu", 2)


def test_json_roundtrip_preserves_accents(tmp_path):
    """Transcripts are full of accents; they must survive the round trip."""
    path = tmp_path / "out.json"
    main._save_json(path, [{"text": "año, cámara, ¿qué?"}])

    assert main._load_json(path)[0]["text"] == "año, cámara, ¿qué?"
    assert "\\u" not in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def test_cmd_record_creates_parent_directory(tmp_path):
    """Recording into a new folder works without a prior mkdir."""
    target = tmp_path / "nueva" / "clase.wav"

    with patch("main.record") as record:
        main.cmd_record(_parse("record", "--output", str(target), "--duration", "5"))

    assert target.parent.is_dir()
    assert record.call_args.args[0] == target
    assert record.call_args.args[1] == 5


def test_cmd_record_passes_selected_microphone(tmp_path):
    """--mic reaches the recorder."""
    with patch("main.record") as record:
        main.cmd_record(
            _parse("record", "--output", str(tmp_path / "a.wav"), "--mic", "2")
        )

    assert record.call_args.kwargs["device"] == "2"


def test_cmd_preprocess_passes_denoise_flag(tmp_path):
    """--no-denoise is forwarded as denoise=False."""
    with patch("main.preprocess") as preprocess:
        main.cmd_preprocess(
            _parse("preprocess", "--input", "a.wav", "--output", "b.wav", "--no-denoise")
        )

    assert preprocess.call_args.kwargs["denoise"] is False


def test_cmd_transcribe_derives_output_path(tmp_path):
    """Without --output the JSON sits next to the audio."""
    audio = tmp_path / "clase.wav"
    audio.touch()

    with patch("main.transcribe", return_value=[{"start": 0.0, "end": 1.0, "text": "hola"}]):
        main.cmd_transcribe(_parse("transcribe", "--input", str(audio)))

    assert (tmp_path / "clase.transcription.json").exists()


def test_cmd_diarize_forwards_speaker_hints(tmp_path):
    """--num-speakers reaches the diarizer, where it improves accuracy."""
    audio = tmp_path / "clase.wav"
    audio.touch()

    with patch("main.diarize", return_value=[]) as diarize:
        main.cmd_diarize(_parse("diarize", "--input", str(audio), "--num-speakers", "3"))

    assert diarize.call_args.kwargs["num_speakers"] == 3


def test_cmd_merge_writes_output_and_stats(tmp_path, capsys):
    """Merging writes JSON and prints the speaker breakdown."""
    diarization = tmp_path / "dia.json"
    transcription = tmp_path / "tr.json"
    diarization.write_text(json.dumps([{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}]))
    transcription.write_text(json.dumps([{"start": 0.0, "end": 9.0, "text": "hola"}]))

    main.cmd_merge(
        _parse("merge", "--diarization", str(diarization), "--transcription", str(transcription))
    )

    assert (tmp_path / "tr.merged.json").exists()
    assert "SPEAKER_00" in capsys.readouterr().out


def test_cmd_clean_targets_the_professor(merged_json):
    """The cleanup is pointed at the identified professor only."""
    with patch("main.clean_transcript", side_effect=lambda segments, **kw: segments) as clean:
        main.cmd_clean(_parse("clean", "--input", str(merged_json), "--yes"))

    assert clean.call_args.kwargs["professor"] == "SPEAKER_00"
    assert merged_json.with_suffix(".cleaned.json").exists()


def test_cmd_clean_respects_professor_none(merged_json):
    """--professor none cleans every speaker."""
    with patch("main.clean_transcript", side_effect=lambda segments, **kw: segments) as clean:
        main.cmd_clean(
            _parse("clean", "--input", str(merged_json), "--professor", "none", "--yes")
        )

    assert clean.call_args.kwargs["professor"] is None


def test_cmd_export_writes_requested_formats(workspace, merged_json):
    """Every requested format lands in the output directory."""
    main.cmd_export(
        _parse(
            "export", "--input", str(merged_json), "--name", "clase",
            "--formats", "txt,professor,md,srt", "--yes",
        )
    )

    for filename in ("clase_completo.txt", "clase_profesor.txt", "clase.md", "clase.srt"):
        assert (config.OUTPUT_DIR / filename).exists()


def test_cmd_export_professor_file_excludes_students(workspace, merged_json):
    """The professor-only export really does drop the student's question."""
    main.cmd_export(
        _parse("export", "--input", str(merged_json), "--name", "c",
               "--formats", "professor", "--yes")
    )

    content = (config.OUTPUT_DIR / "c_profesor.txt").read_text(encoding="utf-8")

    assert "Explicación del profesor." in content
    assert "¿Una pregunta?" not in content


def test_cmd_export_rejects_unknown_format(workspace, merged_json):
    """A typo in --formats fails loudly instead of writing nothing."""
    with pytest.raises(ValueError, match="Unknown output format"):
        main.cmd_export(
            _parse("export", "--input", str(merged_json), "--formats", "profesor", "--yes")
        )


def test_cmd_export_defaults_name_to_input_stem(workspace, merged_json):
    """Without --name the output is named after the input file."""
    main.cmd_export(
        _parse("export", "--input", str(merged_json), "--formats", "txt", "--yes")
    )

    assert (config.OUTPUT_DIR / "merged_completo.txt").exists()


def test_cmd_clear_delegates_to_pipeline():
    """`clear` removes a session's cache."""
    with patch("main.pipeline.clear_session") as clear:
        main.cmd_clear(_parse("clear", "--name", "clase"))

    clear.assert_called_once_with("clase")


def test_cmd_devices_prints_table(capsys):
    """`devices` lists indices so --mic can be used."""
    with patch("main.list_input_devices") as devices:
        devices.return_value = [{"index": 3, "name": "USB mic", "channels": 1, "default": True}]
        main.cmd_devices(_parse("devices"))

    output = capsys.readouterr().out
    assert "USB mic" in output
    assert "(default)" in output


def test_cmd_doctor_survives_a_bare_machine(capsys, monkeypatch):
    """`doctor` reports problems instead of failing on them."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with patch("main.list_input_devices", side_effect=OSError("no PortAudio")), \
         patch("main.urllib.request.urlopen", side_effect=OSError("refused")):
        main.cmd_doctor(_parse("--device", "cpu", "doctor"))

    output = capsys.readouterr().out
    assert "MISSING" in output
    assert "profile" in output


def test_cmd_run_builds_options_from_flags(workspace, tmp_path):
    """Every flag maps onto the pipeline options, with the negations inverted."""
    audio = tmp_path / "clase.wav"
    audio.touch()

    with patch("main.pipeline.run_pipeline") as run:
        main.cmd_run(
            _parse(
                "run", "--name", "clase", "--input", str(audio), "--professor", "auto",
                "--num-speakers", "3", "--no-clean", "--no-denoise", "--resume", "--yes",
                "--formats", "professor,json",
            )
        )

    options = run.call_args.args[0]
    assert options.name == "clase"
    assert options.input_path == audio
    assert options.professor == "auto"
    assert options.num_speakers == 3
    assert options.clean_enabled is False
    assert options.denoise is False
    assert options.diarize_enabled is True
    assert options.resume is True
    assert options.interactive is False
    assert options.formats == ("professor", "json")


def test_cmd_run_validates_formats_before_recording(workspace):
    """A bad format is caught up front, not after an hour of recording."""
    with patch("main.pipeline.run_pipeline") as run:
        with pytest.raises(ValueError, match="Unknown output format"):
            main.cmd_run(_parse("run", "--name", "clase", "--formats", "pdf"))

    run.assert_not_called()


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------

def test_main_dispatches_to_the_command(workspace, monkeypatch):
    """main() routes to the handler for the parsed subcommand.

    The handler is patched in COMMANDS, which is what main() consults —
    patching the module attribute would leave the dispatch table pointing
    at the original function.
    """
    monkeypatch.setattr(sys, "argv", ["main.py", "preprocess", "--input", "a", "--output", "b"])
    handler = MagicMock()

    with patch.dict(main.COMMANDS, {"preprocess": handler}):
        main.main()

    handler.assert_called_once()


def test_main_skips_validation_for_doctor(workspace, monkeypatch):
    """`doctor` must run precisely when the environment is broken."""
    monkeypatch.setattr(sys, "argv", ["main.py", "doctor"])

    with patch("main.validate_environment") as validate, \
         patch.dict(main.COMMANDS, {"doctor": MagicMock()}):
        main.main()

    validate.assert_not_called()


def test_main_validates_for_other_commands(workspace, monkeypatch):
    """Any other command is validated before it runs."""
    monkeypatch.setattr(sys, "argv", ["main.py", "clean", "--input", "a.json"])

    with patch("main.validate_environment") as validate, \
         patch.dict(main.COMMANDS, {"clean": MagicMock()}):
        main.main()

    validate.assert_called_once()


def test_main_reports_interrupt_with_resume_hint(workspace, monkeypatch, caplog):
    """Ctrl+C exits 130 and tells the user how to continue."""
    monkeypatch.setattr(sys, "argv", ["main.py", "preprocess", "--input", "a", "--output", "b"])

    with patch("main.validate_environment"), \
         patch.dict(main.COMMANDS, {"preprocess": MagicMock(side_effect=KeyboardInterrupt)}):
        with caplog.at_level("WARNING"):
            with pytest.raises(SystemExit) as exit_info:
                main.main()

    assert exit_info.value.code == 130
    assert "--resume" in caplog.text


def test_main_turns_expected_errors_into_exit_1(workspace, monkeypatch, caplog):
    """A missing file is an error message, not a traceback."""
    monkeypatch.setattr(sys, "argv", ["main.py", "preprocess", "--input", "a", "--output", "b"])

    with patch("main.validate_environment"), \
         patch.dict(
             main.COMMANDS,
             {"preprocess": MagicMock(side_effect=FileNotFoundError("no such file"))},
         ):
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as exit_info:
                main.main()

    assert exit_info.value.code == 1
    assert "no such file" in caplog.text


def test_main_lets_unexpected_errors_surface(workspace, monkeypatch):
    """Programming errors are not swallowed — they must be debuggable."""
    monkeypatch.setattr(sys, "argv", ["main.py", "preprocess", "--input", "a", "--output", "b"])

    with patch("main.validate_environment"), \
         patch.dict(main.COMMANDS, {"preprocess": MagicMock(side_effect=TypeError("bug"))}):
        with pytest.raises(TypeError):
            main.main()


def test_every_command_has_a_handler_and_requirements():
    """The parser, the dispatch table and the prerequisites cannot drift apart.

    Adding a subcommand without wiring it up would otherwise fail only at
    runtime, with a KeyError.
    """
    subcommand_actions = [
        action for action in main.build_parser()._actions
        if hasattr(action, "choices") and isinstance(action.choices, dict)
    ]
    commands = set(subcommand_actions[0].choices)

    assert commands == set(main.COMMANDS)
    assert commands == set(main.REQUIREMENTS)


def test_every_handler_is_callable():
    """Each dispatch entry is a real function."""
    assert all(callable(handler) for handler in main.COMMANDS.values())


def test_commands_table_holds_the_real_handlers():
    """COMMANDS points at the module's own cmd_* functions.

    main() dispatches through this dict, so an entry pointing elsewhere
    would silently run the wrong command.
    """
    for name, handler in main.COMMANDS.items():
        assert handler is getattr(main, f"cmd_{name}")


def test_validate_environment_reports_missing_ollama(workspace, caplog):
    """The Ollama error names the command that starts it."""
    with patch("main.urllib.request.urlopen", side_effect=OSError("refused")):
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit):
                main.validate_environment({"ollama"})

    assert "ollama serve" in caplog.text


def test_validate_environment_reports_missing_microphone(workspace, caplog):
    """The microphone error is surfaced before recording is attempted."""
    with patch("main.list_input_devices", return_value=[]):
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit):
                main.validate_environment({"microphone"})

    assert "Microphone" in caplog.text


def test_validate_environment_reports_every_problem_at_once(workspace, caplog, monkeypatch):
    """All missing prerequisites are listed, not just the first."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with patch("main.urllib.request.urlopen", side_effect=OSError("refused")), \
         patch("main.list_input_devices", return_value=[]):
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit):
                main.validate_environment({"ollama", "hf_token", "microphone"})

    assert "[3]" in caplog.text


def test_cmd_gui_opens_the_window(monkeypatch):
    """`main.py gui` hands over to the desktop window."""
    fake_gui = MagicMock()
    monkeypatch.setitem(sys.modules, "gui", fake_gui)

    main.cmd_gui(_parse("gui"))

    fake_gui.run.assert_called_once()


def test_cmd_gui_explains_a_missing_tkinter(monkeypatch):
    """Without Tkinter the user gets the install command, not a traceback."""
    monkeypatch.setitem(sys.modules, "gui", None)

    with pytest.raises(RuntimeError, match="python3-tk"):
        main.cmd_gui(_parse("gui"))


def test_gui_command_needs_nothing_upfront():
    """The window checks its own prerequisites and explains them in a dialog."""
    assert main.requirements_for(_parse("gui")) == set()
