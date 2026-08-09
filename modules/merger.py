"""M5 — Merge diarization and transcription.

Diarization knows *who* spoke when; Whisper knows *what* was said when.
Joining them is what produces a professor-only transcript, and doing it
per segment is not good enough: Whisper happily emits a single segment
that ends with the professor's question and continues with a student's
answer. Attributing that whole segment to one speaker drags the other
speaker's words into the output.

So when word timestamps are available, every *word* is attributed
individually and segments are split at speaker boundaries. Isolated
one-word flips — almost always diarization jitter rather than a real
interjection — are smoothed back into their surroundings.
"""

import logging
from bisect import bisect_left, bisect_right

import config

logger = logging.getLogger(__name__)

UNKNOWN_SPEAKER = "UNKNOWN"


class _SpeakerIndex:
    """Answers "who was speaking during this interval?" efficiently.

    Diarization output may contain overlapping segments (two people
    talking at once), so this cannot be a simple ordered walk. Instead it
    binary-searches into the segments sorted by start time and only scans
    the window that could possibly overlap the query.
    """

    def __init__(self, diarization: list[dict]):
        self._segments = sorted(diarization, key=lambda segment: segment["start"])
        self._starts = [segment["start"] for segment in self._segments]
        self._max_duration = max(
            (segment["end"] - segment["start"] for segment in self._segments),
            default=0.0,
        )

    def best_speaker(self, start: float, end: float) -> tuple[str, float]:
        """Returns the speaker with most overlap in `[start, end]`.

        Returns `(UNKNOWN_SPEAKER, 0.0)` when nothing overlaps — which
        happens for words Whisper heard but diarization considered noise.
        """
        if not self._segments:
            return UNKNOWN_SPEAKER, 0.0

        low = bisect_left(self._starts, start - self._max_duration)
        high = bisect_right(self._starts, end)

        totals: dict[str, float] = {}
        for index in range(low, high):
            segment = self._segments[index]
            overlap = min(end, segment["end"]) - max(start, segment["start"])
            if overlap > 0:
                totals[segment["speaker"]] = totals.get(segment["speaker"], 0.0) + overlap

        if not totals:
            return UNKNOWN_SPEAKER, 0.0

        speaker = max(totals, key=totals.get)
        return speaker, totals[speaker]


def _smooth(words: list[dict]) -> list[dict]:
    """Removes isolated single-word speaker flips.

    A lone word attributed to a different speaker, surrounded on both
    sides by the same speaker, is diarization jitter far more often than a
    real interruption. Words with no speaker at all inherit their
    neighbours' when both agree.
    """
    if len(words) < 3:
        return words

    smoothed = 0
    for index in range(1, len(words) - 1):
        previous = words[index - 1]["speaker"]
        following = words[index + 1]["speaker"]
        current = words[index]["speaker"]
        if previous == following and current != previous:
            words[index]["speaker"] = previous
            smoothed += 1

    if smoothed:
        logger.info("Smoothed %d isolated word-level speaker flip(s)", smoothed)
    return words


def _group_words(words: list[dict]) -> list[dict]:
    """Groups consecutive same-speaker words into segments."""
    groups: list[dict] = []
    for word in words:
        text = word["word"]
        if groups and groups[-1]["speaker"] == word["speaker"]:
            groups[-1]["end"] = word["end"]
            groups[-1]["text"] += text
        else:
            groups.append(
                {
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": word["speaker"],
                    "text": text,
                }
            )

    for group in groups:
        group["text"] = group["text"].strip()

    return [group for group in groups if group["text"]]


def merge(
    diarization: list[dict],
    transcription: list[dict],
    use_words: bool = True,
) -> list[dict]:
    """Attaches speaker labels to transcribed text.

    Args:
        diarization: Dicts with keys "start", "end", "speaker".
        transcription: Dicts with keys "start", "end", "text" and
            optionally "words" (from `word_timestamps=True`).
        use_words: When True (default) and word timings are present,
            attribute each word individually and split segments at
            speaker changes. Set False to attribute whole segments.

    Returns:
        A list of dicts sorted by start time:
        `[{"start", "end", "speaker", "text"}, ...]`. Text with no
        overlapping diarization is labelled `UNKNOWN`.

    Raises:
        ValueError: If either input list is empty.
    """
    if not diarization:
        raise ValueError("Diarization segments list is empty.")
    if not transcription:
        raise ValueError("Transcription segments list is empty.")

    index = _SpeakerIndex(diarization)
    word_level = use_words and any(segment.get("words") for segment in transcription)

    merged: list[dict] = []

    if word_level:
        words: list[dict] = []
        for segment in transcription:
            segment_words = segment.get("words")
            if segment_words:
                for word in segment_words:
                    speaker, _ = index.best_speaker(word["start"], word["end"])
                    words.append(
                        {
                            "start": word["start"],
                            "end": word["end"],
                            "word": word["word"],
                            "speaker": speaker,
                        }
                    )
            else:
                # Segment without word timings: keep it whole.
                speaker, _ = index.best_speaker(segment["start"], segment["end"])
                words.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "word": " " + segment["text"].strip(),
                        "speaker": speaker,
                    }
                )

        merged = _group_words(_smooth(words))
        logger.info(
            "Merged %d transcription segments into %d speaker turns using word-level timings",
            len(transcription),
            len(merged),
        )
    else:
        for segment in transcription:
            speaker, _ = index.best_speaker(segment["start"], segment["end"])
            merged.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": speaker,
                    "text": segment["text"].strip(),
                }
            )
        logger.info("Merged %d transcription segments with speaker labels", len(merged))

    merged.sort(key=lambda segment: segment["start"])
    return merged


def merge_turns(
    segments: list[dict],
    max_gap: float = config.TURN_MAX_GAP_SECONDS,
    max_seconds: float = config.TURN_MAX_SECONDS,
) -> list[dict]:
    """Joins consecutive same-speaker segments into readable turns.

    Whisper cuts on breath pauses, which produces a shredded transcript.
    Recombining those fragments gives the LLM cleanup stage whole
    sentences to work with and makes the exported text readable.

    Args:
        segments: Speaker-labelled segments sorted by start time.
        max_gap: Maximum silence, in seconds, that still counts as the
            same turn.
        max_seconds: Never produce a turn longer than this.

    Returns:
        A new list of merged segments.
    """
    if not segments:
        return []

    turns: list[dict] = [dict(segments[0])]
    for segment in segments[1:]:
        current = turns[-1]
        gap = segment["start"] - current["end"]
        duration = segment["end"] - current["start"]
        if (
            segment["speaker"] == current["speaker"]
            and gap <= max_gap
            and duration <= max_seconds
        ):
            current["end"] = segment["end"]
            current["text"] = f"{current['text']} {segment['text']}".strip()
        else:
            turns.append(dict(segment))

    logger.info("Grouped %d segments into %d turns", len(segments), len(turns))
    return turns
