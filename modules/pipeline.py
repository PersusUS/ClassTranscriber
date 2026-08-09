"""Pipeline orchestration with resumable stages.

A two-hour lecture takes a while to process on a laptop, and losing all
of it because Ollama was not running when stage six started is a bad
trade. Every stage therefore writes its result into
`sessions/<name>/` as JSON, and `--resume` picks up from the last stage
that completed. Re-exporting with different options, or re-running just
the LLM cleanup, then costs seconds instead of an hour.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import config
from config import Settings
from modules import exporter
from modules.cleaner import clean_transcript
from modules.diarizer import diarize
from modules.merger import merge, merge_turns
from modules.preprocessor import preprocess
from modules.recorder import record
from modules.speaker_id import resolve_professor, speaker_stats
from modules.transcriber import transcribe

logger = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    """Everything the pipeline needs to know for one session."""

    name: str
    settings: Settings
    input_path: Path | None = None      # Existing recording; None means "record now"
    duration: int | None = None
    device: int | str | None = None     # Audio input device
    professor: str | None = None
    num_speakers: int | None = None
    max_speakers: int = config.MAX_SPEAKERS
    denoise: bool = True
    diarize_enabled: bool = True
    clean_enabled: bool = True
    resume: bool = False
    interactive: bool = True
    hf_token: str | None = None
    formats: tuple[str, ...] = ("txt", "professor")
    keep_audio: bool = True
    extra: dict = field(default_factory=dict)


def session_dir(name: str) -> Path:
    """Returns (and creates) the working directory for a session."""
    path = config.SESSIONS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _cached(path: Path, resume: bool, label: str):
    """Returns a cached stage result when resuming, else None."""
    if resume and path.exists():
        logger.info("Reusing cached %s from %s", label, path.name)
        return _read_json(path)
    return None


def run_pipeline(options: PipelineOptions) -> dict[str, Path]:
    """Runs the full pipeline and returns the paths of every file written.

    Args:
        options: Session configuration.

    Returns:
        A mapping of artifact name to path, e.g.
        `{"professor": Path("output/clase01_profesor.txt"), ...}`.
    """
    workdir = session_dir(options.name)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = workdir / "raw.wav"
    clean_path = workdir / "clean.wav"
    transcription_path = workdir / "transcription.json"
    diarization_path = workdir / "diarization.json"
    merged_path = workdir / "merged.json"
    cleaned_path = workdir / "cleaned.json"

    logger.info("Session '%s' — %s", options.name, options.settings.describe())

    # --- 1. Audio ---------------------------------------------------------
    if options.input_path is not None:
        source = Path(options.input_path)
        if not source.exists():
            raise FileNotFoundError(f"Input audio not found: {source}")
        logger.info("=== Step 1/6: Using existing recording %s ===", source)
        raw_path = source
    elif options.resume and raw_path.exists():
        logger.info("=== Step 1/6: Reusing recording %s ===", raw_path.name)
    else:
        logger.info("=== Step 1/6: Recording ===")
        record(raw_path, options.duration, device=options.device)

    # --- 2. Preprocess ----------------------------------------------------
    if options.resume and clean_path.exists():
        logger.info("=== Step 2/6: Reusing preprocessed audio ===")
    else:
        logger.info("=== Step 2/6: Preprocessing ===")
        preprocess(raw_path, clean_path, denoise=options.denoise)

    # --- 3. Transcribe ----------------------------------------------------
    transcription = _cached(transcription_path, options.resume, "transcription")
    if transcription is None:
        logger.info("=== Step 3/6: Transcribing ===")
        transcription = transcribe(clean_path, settings=options.settings)
        _write_json(transcription_path, transcription)

    if not transcription:
        raise RuntimeError(
            "Whisper produced no usable speech. Check that the recording is not silent "
            "and that the language setting matches the class."
        )

    # --- 4. Diarize and merge --------------------------------------------
    if options.diarize_enabled:
        diarization = _cached(diarization_path, options.resume, "diarization")
        if diarization is None:
            logger.info("=== Step 4/6: Diarizing ===")
            diarization = diarize(
                clean_path,
                hf_token=options.hf_token,
                max_speakers=options.max_speakers,
                num_speakers=options.num_speakers,
                device=options.settings.device,
            )
            _write_json(diarization_path, diarization)

        merged = merge_turns(merge(diarization, transcription))
    else:
        logger.info("=== Step 4/6: Diarization disabled — treating all speech as one speaker ===")
        merged = merge_turns(
            [
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": "SPEAKER_00",
                    "text": segment["text"],
                }
                for segment in transcription
            ]
        )

    _write_json(merged_path, merged)

    # --- 5. Identify the professor and clean ------------------------------
    professor = resolve_professor(
        merged,
        requested=options.professor,
        interactive=options.interactive,
    )

    segments = merged
    if options.clean_enabled:
        cached_clean = _cached(cleaned_path, options.resume, "cleaned transcript")
        if cached_clean is not None:
            segments = cached_clean
        else:
            logger.info("=== Step 5/6: Cleaning with the local LLM ===")
            segments = clean_transcript(
                [dict(segment) for segment in merged],
                model=options.settings.ollama_model,
                professor=professor,
                language=options.settings.language,
            )
            _write_json(cleaned_path, segments)
    else:
        logger.info("=== Step 5/6: LLM cleanup disabled ===")

    # --- 6. Export --------------------------------------------------------
    logger.info("=== Step 6/6: Exporting ===")
    outputs = _export_all(segments, options, professor)

    if not options.keep_audio and options.input_path is None:
        raw_path.unlink(missing_ok=True)
        logger.info("Raw recording deleted (--no-keep-audio)")

    logger.info("Session '%s' complete", options.name)
    for label, path in outputs.items():
        logger.info("  %-12s %s", label, path)
    return outputs


def _export_all(
    segments: list[dict],
    options: PipelineOptions,
    professor: str | None,
) -> dict[str, Path]:
    """Writes every requested output format, plus the speaker breakdown."""
    outputs = exporter.write_outputs(
        segments,
        name=options.name,
        output_dir=config.OUTPUT_DIR,
        formats=options.formats,
        professor=professor,
    )

    stats_path = session_dir(options.name) / "speakers.json"
    _write_json(stats_path, speaker_stats(segments))
    outputs["speakers"] = stats_path

    return outputs


def clear_session(name: str) -> None:
    """Deletes a session's cached intermediate files."""
    path = config.SESSIONS_DIR / name
    if path.exists():
        shutil.rmtree(path)
        logger.info("Removed session cache %s", path)
