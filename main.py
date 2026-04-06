"""ClassTranscriber — CLI entry point.

Record classroom audio, separate speakers via diarization, transcribe
with Whisper, and produce a clean .txt file per class session corrected
by a local LLM.
"""

import argparse
import json
import logging
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

import torch
from dotenv import load_dotenv

import config
from modules.recorder import record
from modules.preprocessor import preprocess
from modules.diarizer import diarize
from modules.transcriber import transcribe
from modules.merger import merge
from modules.cleaner import clean_transcript
from modules.exporter import export

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """Checks all required environment prerequisites before running any command.

    Validates:
        - .env file exists with instructions to create it if missing
        - HF_TOKEN is set and starts with 'hf_'
        - CUDA is available via torch
        - Ollama is reachable at http://localhost:11434
        - audio/ and output/ directories exist (creates them if not)

    Raises:
        SystemExit: If any critical check fails.
    """
    errors = []

    # 1. .env file
    env_path = config.BASE_DIR / ".env"
    if not env_path.exists():
        errors.append(
            ".env file not found. Create it by running:\n"
            "  copy .env.example .env\n"
            "Then edit .env and set your HuggingFace token:\n"
            "  HF_TOKEN=hf_your_token_here\n"
            "Get a token at: https://huggingface.co/settings/tokens"
        )

    # 2. HF_TOKEN
    if not HF_TOKEN:
        errors.append(
            "HF_TOKEN is not set. Add it to your .env file:\n"
            "  HF_TOKEN=hf_your_token_here"
        )
    elif not HF_TOKEN.startswith("hf_"):
        errors.append(
            f"HF_TOKEN appears invalid (expected 'hf_...' but got '{HF_TOKEN[:6]}...'). "
            "Get a valid token at: https://huggingface.co/settings/tokens"
        )

    # 3. CUDA
    if not torch.cuda.is_available():
        errors.append(
            "CUDA is not available. ClassTranscriber requires an NVIDIA GPU with CUDA 12.\n"
            "Ensure you have the correct NVIDIA drivers and PyTorch with CUDA installed:\n"
            "  pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121"
        )

    # 4. Ollama
    try:
        req = urllib.request.Request("http://localhost:11434", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            pass
    except (urllib.error.URLError, OSError):
        errors.append(
            "Ollama is not reachable at http://localhost:11434.\n"
            "Run: ollama serve"
        )

    # 5. Directories
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if errors:
        logger.error("Environment validation failed:")
        for i, err in enumerate(errors, 1):
            logger.error("  [%d] %s", i, err)
        sys.exit(1)

    logger.info("Environment validation passed")


def cmd_record(args: argparse.Namespace) -> None:
    """Handle the 'record' subcommand."""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record(output_path, args.duration)
    logger.info("Recording complete: %s", output_path)


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Handle the 'preprocess' subcommand."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocess(input_path, output_path)
    logger.info("Preprocessing complete: %s", output_path)


def cmd_diarize(args: argparse.Namespace) -> None:
    """Handle the 'diarize' subcommand."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Copy .env.example to .env and fill in your token.")
    input_path = Path(args.input)
    segments = diarize(input_path, hf_token=HF_TOKEN)
    for seg in segments:
        logger.info("[%.1fs -> %.1fs] %s", seg["start"], seg["end"], seg["speaker"])
    logger.info("Diarization complete: %d segments", len(segments))


def cmd_transcribe(args: argparse.Namespace) -> None:
    """Handle the 'transcribe' subcommand."""
    input_path = Path(args.input)
    segments = transcribe(input_path)
    for seg in segments:
        logger.info("[%.2fs -> %.2fs] %s", seg["start"], seg["end"], seg["text"])
    logger.info("Transcription complete: %d segments", len(segments))


def cmd_clean(args: argparse.Namespace) -> None:
    """Handle the 'clean' subcommand."""
    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    cleaned = clean_transcript(segments)
    # Save cleaned segments back
    output_path = input_path.with_suffix(".cleaned.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    logger.info("Cleaning complete: %s", output_path)


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand — full pipeline end-to-end."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Copy .env.example to .env and fill in your token.")

    name = args.name
    raw_path = config.AUDIO_DIR / f"{name}_raw.wav"
    clean_path = config.AUDIO_DIR / f"{name}_clean.wav"
    output_path = config.OUTPUT_DIR / f"{name}.txt"

    # Ensure directories exist
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Record
    logger.info("=== Step 1/7: Recording ===")
    record(raw_path, args.duration)

    # 2. Preprocess
    logger.info("=== Step 2/7: Preprocessing ===")
    preprocess(raw_path, clean_path)

    # 3. Diarize
    logger.info("=== Step 3/7: Diarizing ===")
    diarization_segments = diarize(clean_path, hf_token=HF_TOKEN)
    for seg in diarization_segments:
        logger.info("[%.1fs -> %.1fs] %s", seg["start"], seg["end"], seg["speaker"])

    # 4. Transcribe
    logger.info("=== Step 4/7: Transcribing ===")
    transcription_segments = transcribe(clean_path)

    # 5. Merge
    logger.info("=== Step 5/7: Merging ===")
    merged_segments = merge(diarization_segments, transcription_segments)

    # 5b. Save raw (uncleaned) transcript
    raw_transcript_path = config.OUTPUT_DIR / f"{name}_raw.txt"
    export(merged_segments, raw_transcript_path)
    logger.info("Raw transcript saved: %s", raw_transcript_path)

    # 6. Clean
    logger.info("=== Step 6/7: Cleaning with LLM ===")
    cleaned_segments = clean_transcript(merged_segments)

    # 7. Export
    logger.info("=== Step 7/7: Exporting ===")
    export(cleaned_segments, output_path)

    logger.info("Pipeline complete! Output: %s", output_path)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        prog="ClassTranscriber",
        description="Record, diarize, transcribe, and clean classroom audio.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # record
    p_record = subparsers.add_parser("record", help="Record audio from microphone")
    p_record.add_argument("--duration", type=int, default=config.DEFAULT_DURATION, help="Duration in seconds")
    p_record.add_argument("--output", type=str, required=True, help="Output WAV file path")

    # preprocess
    p_preprocess = subparsers.add_parser("preprocess", help="Noise reduction and normalization")
    p_preprocess.add_argument("--input", type=str, required=True, help="Input WAV file path")
    p_preprocess.add_argument("--output", type=str, required=True, help="Output WAV file path")

    # diarize
    p_diarize = subparsers.add_parser("diarize", help="Speaker diarization")
    p_diarize.add_argument("--input", type=str, required=True, help="Input WAV file path")

    # transcribe
    p_transcribe = subparsers.add_parser("transcribe", help="Whisper transcription")
    p_transcribe.add_argument("--input", type=str, required=True, help="Input WAV file path")

    # clean
    p_clean = subparsers.add_parser("clean", help="LLM transcript cleanup")
    p_clean.add_argument("--input", type=str, required=True, help="Input JSON file with merged segments")

    # run (full pipeline)
    p_run = subparsers.add_parser("run", help="Full end-to-end pipeline")
    p_run.add_argument("--duration", type=int, default=config.DEFAULT_DURATION, help="Recording duration in seconds")
    p_run.add_argument("--name", type=str, required=True, help="Session name (used for filenames)")

    args = parser.parse_args()

    validate_environment()

    commands = {
        "record": cmd_record,
        "preprocess": cmd_preprocess,
        "diarize": cmd_diarize,
        "transcribe": cmd_transcribe,
        "clean": cmd_clean,
        "run": cmd_run,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
