"""ClassTranscriber — CLI entry point.

Records a university lecture, separates the speakers, transcribes it with
Whisper and produces a clean transcript of the professor alone, ready to
turn into study notes. Everything runs locally.

Typical use:

    python main.py run --name algebra_01 --duration 5400

Each stage is also available on its own, and `--resume` reuses whatever
the session already computed.
"""

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

import config
from modules import exporter, pipeline
from modules.cleaner import clean_transcript
from modules.diarizer import diarize
from modules.merger import merge, merge_turns
from modules.preprocessor import preprocess
from modules.recorder import list_input_devices, record
from modules.speaker_id import format_stats, resolve_professor, speaker_stats
from modules.transcriber import transcribe

load_dotenv()

logger = logging.getLogger("classtranscriber")

# What each command actually needs, so that `record` does not fail because
# Ollama is not running and `transcribe` does not demand a HuggingFace token.
REQUIREMENTS = {
    "record": {"microphone"},
    "preprocess": set(),
    "transcribe": set(),
    "diarize": {"hf_token"},
    "merge": set(),
    "clean": {"ollama"},
    "export": set(),
    "run": set(),          # Computed dynamically from the flags
    "devices": {"microphone"},
    "doctor": set(),
    "clear": set(),
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def check_ollama() -> tuple[bool, str]:
    """Reports whether the Ollama server is reachable."""
    try:
        request = urllib.request.Request(config.OLLAMA_HOST, method="GET")
        with urllib.request.urlopen(request, timeout=5):
            return True, f"reachable at {config.OLLAMA_HOST}"
    except (urllib.error.URLError, OSError) as exc:
        return False, f"not reachable at {config.OLLAMA_HOST} ({exc})"


def check_hf_token() -> tuple[bool, str]:
    """Reports whether a plausible HuggingFace token is configured."""
    token = os.getenv("HF_TOKEN")
    if not token:
        return False, "HF_TOKEN is not set (needed for speaker diarization)"
    if not token.startswith("hf_"):
        return False, f"HF_TOKEN looks invalid (starts with '{token[:4]}...')"
    return True, "configured"


def check_microphone() -> tuple[bool, str]:
    """Reports whether at least one input device exists."""
    try:
        devices = list_input_devices()
    except Exception as exc:      # noqa: BLE001 - PortAudio failures vary by OS
        return False, f"could not query audio devices ({exc})"
    if not devices:
        return False, "no input device found"
    default = next((device for device in devices if device["default"]), devices[0])
    return True, f"{len(devices)} input device(s), default: {default['name']}"


def validate_environment(requires: set[str]) -> None:
    """Fails fast when a prerequisite for the chosen command is missing."""
    errors = []

    if "ollama" in requires:
        ok, detail = check_ollama()
        if not ok:
            errors.append(f"Ollama {detail}.\n      Start it with: ollama serve")

    if "hf_token" in requires:
        ok, detail = check_hf_token()
        if not ok:
            errors.append(
                f"{detail}.\n"
                "      1. Accept https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "      2. Create a read token at https://huggingface.co/settings/tokens\n"
                "      3. Add HF_TOKEN=hf_... to .env  (cp .env.example .env)"
            )

    if "microphone" in requires:
        ok, detail = check_microphone()
        if not ok:
            errors.append(f"Microphone: {detail}")

    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    if errors:
        logger.error("Cannot run this command:")
        for index, error in enumerate(errors, 1):
            logger.error("  [%d] %s", index, error)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def settings_from_args(args: argparse.Namespace):
    """Builds resolved settings from the global CLI flags."""
    return config.resolve_settings(
        profile=getattr(args, "profile", None),
        language=getattr(args, "language", None),
        device=getattr(args, "device", None),
        cpu_threads=getattr(args, "threads", None),
        whisper_model=getattr(args, "model", None),
    )


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_devices(args: argparse.Namespace) -> None:
    """Lists the available microphones."""
    devices = list_input_devices()
    print(f"{'IDX':<5} {'CH':<4} NAME")
    for device in devices:
        marker = " (default)" if device["default"] else ""
        print(f"{device['index']:<5} {device['channels']:<4} {device['name']}{marker}")
    print("\nUse --mic <IDX> with `record` or `run` to choose one.")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Prints a full environment report without failing."""
    settings = settings_from_args(args)

    print("ClassTranscriber environment\n" + "=" * 40)
    print(f"Resolved settings : {settings.describe()}")
    print(f"CUDA available    : {config.cuda_available()}"
          f"{f' ({config.gpu_vram_gb():.1f} GB VRAM)' if config.cuda_available() else ''}")
    print(f"CPU threads       : {settings.cpu_threads} of {os.cpu_count()}")

    for label, (ok, detail) in {
        "Microphone": check_microphone(),
        "Ollama": check_ollama(),
        "HuggingFace": check_hf_token(),
    }.items():
        print(f"{label:<18}: {'OK' if ok else 'MISSING'} — {detail}")

    for name, profile in config.PROFILES.items():
        active = " <- active" if name == settings.profile.name else ""
        print(f"  profile {name:<9} whisper={profile.whisper_model:<16} "
              f"llm={profile.ollama_model:<12}{active}")

    print("\nNote: diarization is only needed to isolate the professor. "
          "Run with --no-diarize to transcribe everything without a HuggingFace token.")


def cmd_record(args: argparse.Namespace) -> None:
    """Records audio to a WAV file."""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record(output_path, args.duration, device=args.mic)


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Denoises and normalises a recording."""
    preprocess(Path(args.input), Path(args.output), denoise=not args.no_denoise)


def cmd_transcribe(args: argparse.Namespace) -> None:
    """Transcribes a recording and saves the segments as JSON."""
    input_path = Path(args.input)
    segments = transcribe(input_path, settings=settings_from_args(args))
    output = Path(args.output) if args.output else input_path.with_suffix(".transcription.json")
    _save_json(output, segments)
    logger.info("Transcription saved to %s (%d segments)", output, len(segments))


def cmd_diarize(args: argparse.Namespace) -> None:
    """Diarizes a recording and saves the segments as JSON."""
    segments = diarize(
        Path(args.input),
        hf_token=os.getenv("HF_TOKEN"),
        max_speakers=args.max_speakers,
        num_speakers=args.num_speakers,
        device=settings_from_args(args).device,
    )
    output = Path(args.output) if args.output else Path(args.input).with_suffix(".diarization.json")
    _save_json(output, segments)
    logger.info("Diarization saved to %s (%d segments)", output, len(segments))


def cmd_merge(args: argparse.Namespace) -> None:
    """Merges diarization and transcription JSON files."""
    transcription_path = Path(args.transcription)
    merged = merge_turns(
        merge(_load_json(Path(args.diarization)), _load_json(transcription_path))
    )
    output = Path(args.output) if args.output else transcription_path.with_suffix(".merged.json")
    _save_json(output, merged)
    print(format_stats(speaker_stats(merged)))
    logger.info("Merged transcript saved to %s (%d turns)", output, len(merged))


def cmd_clean(args: argparse.Namespace) -> None:
    """Runs the LLM cleanup on a merged transcript."""
    input_path = Path(args.input)
    segments = _load_json(input_path)
    settings = settings_from_args(args)

    professor = resolve_professor(segments, requested=args.professor, interactive=not args.yes)
    cleaned = clean_transcript(
        segments,
        model=settings.ollama_model,
        professor=professor,
        language=settings.language,
    )

    output = Path(args.output) if args.output else input_path.with_suffix(".cleaned.json")
    _save_json(output, cleaned)
    logger.info("Cleaned transcript saved to %s", output)


def cmd_export(args: argparse.Namespace) -> None:
    """Exports a merged or cleaned transcript to readable files."""
    input_path = Path(args.input)
    segments = _load_json(input_path)
    formats = exporter.parse_formats(args.formats)
    professor = resolve_professor(segments, requested=args.professor, interactive=not args.yes)

    exporter.write_outputs(
        segments,
        name=args.name or input_path.stem,
        output_dir=config.OUTPUT_DIR,
        formats=formats,
        professor=professor,
    )


def cmd_clear(args: argparse.Namespace) -> None:
    """Deletes a session's cached intermediate files."""
    pipeline.clear_session(args.name)


def cmd_run(args: argparse.Namespace) -> None:
    """Runs the whole pipeline for one class."""
    options = pipeline.PipelineOptions(
        name=args.name,
        settings=settings_from_args(args),
        input_path=Path(args.input) if args.input else None,
        duration=args.duration,
        device=args.mic,
        professor=args.professor,
        num_speakers=args.num_speakers,
        max_speakers=args.max_speakers,
        denoise=not args.no_denoise,
        diarize_enabled=not args.no_diarize,
        clean_enabled=not args.no_clean,
        resume=args.resume,
        interactive=not args.yes,
        hf_token=os.getenv("HF_TOKEN"),
        formats=exporter.parse_formats(args.formats),
        keep_audio=not args.no_keep_audio,
    )
    pipeline.run_pipeline(options)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="ClassTranscriber",
        description="Record, diarize, transcribe and clean university lectures — locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        default=config.DEFAULT_PROFILE,
        help="Resource profile: low, balanced, quality or auto",
    )
    parser.add_argument("--language", default=config.LANGUAGE, help="ISO 639-1 language code")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu or cuda")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for inference")
    parser.add_argument("--model", default=None, help="Override the profile's Whisper model")
    parser.add_argument("--log-level", default=config.LOG_LEVEL, help="DEBUG, INFO, WARNING…")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("devices", help="List available microphones")
    subparsers.add_parser("doctor", help="Check the environment and show resolved settings")

    p_record = subparsers.add_parser("record", help="Record audio from a microphone")
    p_record.add_argument("--duration", type=int, default=None,
                          help="Seconds to record (omit to record until Ctrl+C)")
    p_record.add_argument("--output", required=True, help="Output WAV path")
    p_record.add_argument("--mic", default=None, help="Input device index or name")

    p_preprocess = subparsers.add_parser("preprocess", help="Denoise and normalise a recording")
    p_preprocess.add_argument("--input", required=True)
    p_preprocess.add_argument("--output", required=True)
    p_preprocess.add_argument("--no-denoise", action="store_true", help="Skip spectral gating")

    p_transcribe = subparsers.add_parser("transcribe", help="Transcribe a recording")
    p_transcribe.add_argument("--input", required=True)
    p_transcribe.add_argument("--output", default=None, help="Output JSON path")

    p_diarize = subparsers.add_parser("diarize", help="Separate speakers")
    p_diarize.add_argument("--input", required=True)
    p_diarize.add_argument("--output", default=None, help="Output JSON path")
    p_diarize.add_argument("--num-speakers", type=int, default=None,
                           help="Exact number of speakers, when known")
    p_diarize.add_argument("--max-speakers", type=int, default=config.MAX_SPEAKERS)

    p_merge = subparsers.add_parser("merge", help="Combine diarization and transcription")
    p_merge.add_argument("--diarization", required=True, help="Diarization JSON")
    p_merge.add_argument("--transcription", required=True, help="Transcription JSON")
    p_merge.add_argument("--output", default=None)

    p_clean = subparsers.add_parser("clean", help="LLM cleanup of a merged transcript")
    p_clean.add_argument("--input", required=True, help="Merged segments JSON")
    p_clean.add_argument("--output", default=None)
    p_clean.add_argument("--professor", default=None,
                         help="Speaker label, rank index, 'auto' or 'none'")
    p_clean.add_argument("--yes", action="store_true", help="Never prompt; accept the top speaker")

    p_export = subparsers.add_parser("export", help="Write transcripts from a JSON file")
    p_export.add_argument("--input", required=True)
    p_export.add_argument("--name", default=None, help="Base name for the output files")
    p_export.add_argument("--professor", default=None)
    p_export.add_argument("--formats", default="txt,professor,md",
                          help="Comma-separated: txt, professor, md, srt")
    p_export.add_argument("--yes", action="store_true")

    p_clear = subparsers.add_parser("clear", help="Delete a session's cached files")
    p_clear.add_argument("--name", required=True)

    p_run = subparsers.add_parser("run", help="Full pipeline for one class")
    p_run.add_argument("--name", required=True, help="Session name, used for filenames")
    p_run.add_argument("--input", default=None,
                       help="Process an existing recording instead of recording now")
    p_run.add_argument("--duration", type=int, default=config.DEFAULT_DURATION,
                       help="Recording length in seconds (Ctrl+C stops early)")
    p_run.add_argument("--mic", default=None, help="Input device index or name")
    p_run.add_argument("--professor", default=None,
                       help="Speaker label, rank index, 'auto' or 'none'")
    p_run.add_argument("--num-speakers", type=int, default=None)
    p_run.add_argument("--max-speakers", type=int, default=config.MAX_SPEAKERS)
    p_run.add_argument("--no-denoise", action="store_true")
    p_run.add_argument("--no-diarize", action="store_true",
                       help="Skip speaker separation (no HuggingFace token needed)")
    p_run.add_argument("--no-clean", action="store_true", help="Skip the LLM cleanup")
    p_run.add_argument("--resume", action="store_true",
                       help="Reuse whatever this session already computed")
    p_run.add_argument("--yes", action="store_true", help="Never prompt")
    p_run.add_argument("--formats", default="txt,professor,md",
                       help="Comma-separated: txt, professor, md, srt, json")
    p_run.add_argument("--no-keep-audio", action="store_true",
                       help="Delete the raw recording once the transcript is written")

    return parser


# Command name -> handler. Kept beside REQUIREMENTS so the two stay in step.
COMMANDS = {
    "devices": cmd_devices,
    "doctor": cmd_doctor,
    "record": cmd_record,
    "preprocess": cmd_preprocess,
    "transcribe": cmd_transcribe,
    "diarize": cmd_diarize,
    "merge": cmd_merge,
    "clean": cmd_clean,
    "export": cmd_export,
    "clear": cmd_clear,
    "run": cmd_run,
}


def requirements_for(args: argparse.Namespace) -> set[str]:
    """Works out which prerequisites the invocation actually needs."""
    requires = set(REQUIREMENTS.get(args.command, set()))

    if args.command == "run":
        if not args.input:
            requires.add("microphone")
        if not args.no_diarize:
            requires.add("hf_token")
        if not args.no_clean:
            requires.add("ollama")

    return requires


def main() -> None:
    """Parses arguments, validates the environment and dispatches."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format=config.LOG_FORMAT)

    if args.command != "doctor":
        validate_environment(requirements_for(args))

    try:
        COMMANDS[args.command](args)
    except KeyboardInterrupt:
        logger.warning("Interrupted. Re-run with --resume to continue from here.")
        sys.exit(130)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
