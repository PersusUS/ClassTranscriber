"""M5 — Merge diarization and transcription segments.

Aligns diarization segments (who spoke when) with transcription segments
(what was said when) to produce speaker-attributed text.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def merge(diarization: list[dict], transcription: list[dict]) -> list[dict]:
    """Merges diarization and transcription segments by time overlap.

    For each transcription segment, assigns the speaker whose diarization
    segment has the maximum overlap with that transcription segment's time
    window.

    Args:
        diarization: List of dicts with keys "start", "end", "speaker".
        transcription: List of dicts with keys "start", "end", "text".

    Returns:
        A list of dicts sorted by start time:
        [{"start": float, "end": float, "speaker": str, "text": str}, ...]
        If no diarization segment overlaps a transcription segment,
        speaker is "UNKNOWN".

    Raises:
        ValueError: If either input list is empty.
    """
    if not diarization:
        raise ValueError("Diarization segments list is empty.")
    if not transcription:
        raise ValueError("Transcription segments list is empty.")

    merged = []

    for tseg in transcription:
        ts, te = tseg["start"], tseg["end"]
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for dseg in diarization:
            ds, de = dseg["start"], dseg["end"]
            overlap = min(te, de) - max(ts, ds)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg["speaker"]

        merged.append({
            "start": ts,
            "end": te,
            "speaker": best_speaker,
            "text": tseg["text"],
        })

    merged.sort(key=lambda s: s["start"])

    logger.info("Merged %d transcription segments with speaker labels", len(merged))
    return merged
