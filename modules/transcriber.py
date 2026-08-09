"""M4 — Whisper transcription module.

Wraps faster-whisper with settings chosen for noisy lecture halls and for
machines without a GPU:

* the model size and quantisation come from the active profile, so the
  same code runs on `small`/int8 on a laptop and `large-v3`/float16 on a
  GPU, falling back automatically if a model does not fit;
* `condition_on_previous_text` is off, because in a noisy room feeding
  Whisper its own previous output is what triggers those infinite
  repetition loops;
* VAD trims the silence Whisper would otherwise hallucinate over, and a
  post-filter removes the handful of phrases it invents anyway
  ("Subtítulos realizados por la comunidad de Amara.org" and friends);
* word timestamps are on by default, which is what lets the merger keep
  student questions out of the professor's transcript.
"""

import logging
import time
from pathlib import Path

import config
from config import Settings, resolve_settings

logger = logging.getLogger(__name__)

# Cache of loaded models keyed by configuration, so that running several
# stages in one process does not reload gigabytes of weights.
_models: dict[tuple, object] = {}


def _load_model(settings: Settings):
    """Loads (and caches) a Whisper model, degrading gracefully.

    Tries the requested device and quantisation first, then int8 on the
    same device, then CPU int8. A laptop that runs out of VRAM mid-term
    should still produce a transcript.
    """
    from faster_whisper import WhisperModel

    key = (settings.whisper_model, settings.device, settings.compute_type, settings.cpu_threads)
    if key in _models:
        return _models[key]

    attempts: list[tuple[str, str]] = [(settings.device, settings.compute_type)]
    if settings.device == "cuda":
        attempts.append(("cuda", "int8_float16"))
    attempts.append(("cpu", "int8"))

    last_error: Exception | None = None
    for device, compute_type in attempts:
        logger.info(
            "Loading Whisper '%s' on %s (%s, %d threads)",
            settings.whisper_model,
            device,
            compute_type,
            settings.cpu_threads,
        )
        try:
            started = time.monotonic()
            model = WhisperModel(
                settings.whisper_model,
                device=device,
                compute_type=compute_type,
                cpu_threads=settings.cpu_threads,
            )
            logger.info("Model ready in %.1f s", time.monotonic() - started)
            _models[key] = model
            return model
        except Exception as exc:      # noqa: BLE001 - any load failure is worth retrying
            last_error = exc
            logger.warning(
                "Could not load on %s (%s): %s — trying next fallback",
                device,
                compute_type,
                exc,
            )

    raise RuntimeError(
        f"Could not load Whisper model '{settings.whisper_model}' with any configuration"
    ) from last_error


def is_hallucination(text: str) -> bool:
    """Reports whether a segment matches a known Whisper hallucination.

    These phrases come from subtitled video in Whisper's training data and
    appear over silence or noise, never in an actual lecture.
    """
    normalised = text.strip().lower().rstrip(".!¡ ")
    if not normalised:
        return True
    return any(pattern in normalised for pattern in config.HALLUCINATION_PATTERNS)


def _filter_segments(segments: list[dict]) -> list[dict]:
    """Drops hallucinated, silent and runaway-repetition segments.

    Whisper signals its own uncertainty through `avg_logprob` and
    `no_speech_prob`; in a noisy room those are the cheapest reliable
    filter available. Consecutive identical segments are the classic
    symptom of a decoding loop, so only the first is kept.
    """
    kept: list[dict] = []
    dropped_quality = 0
    dropped_hallucination = 0
    dropped_repetition = 0
    repeats = 0

    for segment in segments:
        text = segment["text"].strip()

        if is_hallucination(text):
            dropped_hallucination += 1
            continue

        if (
            segment.get("avg_logprob") is not None
            and segment["avg_logprob"] < config.MIN_SEGMENT_LOGPROB
        ) or (
            segment.get("no_speech_prob") is not None
            and segment["no_speech_prob"] > config.MAX_SEGMENT_NO_SPEECH_PROB
        ):
            dropped_quality += 1
            continue

        if kept and kept[-1]["text"].strip().lower() == text.lower():
            repeats += 1
            if repeats >= 2:
                dropped_repetition += 1
                continue
        else:
            repeats = 0

        kept.append(segment)

    if dropped_hallucination or dropped_quality or dropped_repetition:
        logger.info(
            "Filtered %d hallucinated, %d low-confidence and %d repeated segment(s)",
            dropped_hallucination,
            dropped_quality,
            dropped_repetition,
        )
    return kept


def transcribe(
    audio_path: Path,
    language: str | None = None,
    settings: Settings | None = None,
    initial_prompt: str | None = None,
    word_timestamps: bool | None = None,
) -> list[dict]:
    """Transcribes an audio file with faster-whisper.

    Args:
        audio_path: Path to the audio file to transcribe.
        language: ISO 639-1 code. Defaults to the configured language ("es").
        settings: Resolved runtime settings. Built from the active profile
            when omitted.
        initial_prompt: Text used to bias vocabulary and punctuation.
            Defaults to the language's lecture prompt from config.
        word_timestamps: Whether to emit per-word timings. Defaults to
            `config.WORD_TIMESTAMPS`.

    Returns:
        A list of dicts with keys "start", "end", "text", "avg_logprob",
        "no_speech_prob" and, when enabled, "words" — a list of
        `{"start", "end", "word"}` dicts.

    Raises:
        FileNotFoundError: If audio_path does not exist.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    settings = settings or resolve_settings(language=language)
    if language:
        settings.language = language.lower()
    if word_timestamps is None:
        word_timestamps = config.WORD_TIMESTAMPS

    model = _load_model(settings)

    logger.info("Transcribing %s (%s)", audio_path.name, settings.describe())

    raw_segments, info = model.transcribe(
        str(audio_path),
        language=settings.language,
        beam_size=settings.beam_size,
        initial_prompt=initial_prompt if initial_prompt is not None else settings.initial_prompt,
        condition_on_previous_text=config.CONDITION_ON_PREVIOUS_TEXT,
        temperature=list(config.TEMPERATURE_FALLBACK),
        no_speech_threshold=config.NO_SPEECH_THRESHOLD,
        log_prob_threshold=config.LOG_PROB_THRESHOLD,
        compression_ratio_threshold=config.COMPRESSION_RATIO_THRESHOLD,
        word_timestamps=word_timestamps,
        vad_filter=config.VAD_FILTER,
        vad_parameters={
            "threshold": config.VAD_THRESHOLD,
            "min_speech_duration_ms": config.VAD_MIN_SPEECH_MS,
            "min_silence_duration_ms": config.VAD_MIN_SILENCE_MS,
            "speech_pad_ms": config.VAD_SPEECH_PAD_MS,
        },
    )

    total_duration = getattr(info, "duration", None)
    logger.info(
        "Detected language: %s (%.1f%% confidence)%s",
        getattr(info, "language", settings.language),
        getattr(info, "language_probability", 0.0) * 100,
        f", audio duration {total_duration:.0f} s" if total_duration else "",
    )

    segments = _consume(raw_segments, total_duration, word_timestamps)
    segments = _filter_segments(segments)

    logger.info("Transcription complete: %d segments kept", len(segments))
    return segments


def _consume(raw_segments, total_duration: float | None, word_timestamps: bool) -> list[dict]:
    """Drains faster-whisper's lazy generator, logging progress as it goes.

    Transcription only actually happens while this generator is consumed,
    so this is the only place that can report progress. On CPU a two-hour
    lecture takes a while, and a silent terminal is indistinguishable
    from a hang.
    """
    segments: list[dict] = []
    started = time.monotonic()
    next_report = 60.0

    for segment in raw_segments:
        entry = {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            "avg_logprob": getattr(segment, "avg_logprob", None),
            "no_speech_prob": getattr(segment, "no_speech_prob", None),
        }

        if word_timestamps and getattr(segment, "words", None):
            entry["words"] = [
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": word.word,
                }
                for word in segment.words
                if word.start is not None and word.end is not None
            ]

        segments.append(entry)

        if entry["end"] >= next_report:
            elapsed = time.monotonic() - started
            speed = entry["end"] / elapsed if elapsed > 0 else 0.0
            if total_duration and speed > 0:
                remaining = max(0.0, total_duration - entry["end"]) / speed
                logger.info(
                    "Transcribed %.0f / %.0f s (%.1fx realtime, ~%.0f s left)",
                    entry["end"],
                    total_duration,
                    speed,
                    remaining,
                )
            else:
                logger.info("Transcribed %.0f s (%.1fx realtime)", entry["end"], speed)
            next_report = entry["end"] + 60.0

    return segments
