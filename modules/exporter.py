"""M7 — Export final transcript to a .txt file.

Writes the cleaned transcript with speaker labels and timestamps
to a human-readable plaintext file.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Formats a float timestamp in seconds to HH:MM:SS.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string like "00:05:23".
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def export(
    segments: list[dict],
    output_path: Path,
    include_timestamps: bool = True,
) -> Path:
    """Writes the transcript to a .txt file.

    Format per line: [HH:MM:SS] SPEAKER_00: text
    If include_timestamps is False, omits the timestamp prefix.
    Adds a blank line between different speaker turns.

    Args:
        segments: List of dicts with keys "start", "end", "speaker", "text".
        output_path: Destination path for the .txt file.
        include_timestamps: Whether to include [HH:MM:SS] prefixes.

    Returns:
        The path to the saved file.

    Raises:
        ValueError: If segments is empty.
    """
    if not segments:
        raise ValueError("Segments list is empty. Nothing to export.")

    header = f"# ClassTranscriber — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")

        prev_speaker = None
        for seg in segments:
            # Add blank line on speaker change
            if prev_speaker is not None and seg["speaker"] != prev_speaker:
                f.write("\n")

            if include_timestamps:
                timestamp = _format_timestamp(seg["start"])
                line = f"[{timestamp}] {seg['speaker']}: {seg['text']}"
            else:
                line = f"{seg['speaker']}: {seg['text']}"

            f.write(line + "\n")
            prev_speaker = seg["speaker"]

    logger.info("Transcript exported to %s (%d segments)", output_path, len(segments))
    return output_path
