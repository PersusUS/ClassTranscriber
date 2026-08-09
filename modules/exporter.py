"""M7 — Export the transcript in several shapes.

Three audiences, three files:

* `export` writes the full speaker-labelled transcript — the record of
  what happened in the room.
* `export_professor` writes only the professor's words as continuous
  prose, with no labels and sparse time markers. This is the file to feed
  to an LLM for summaries and study notes: no student chatter, no
  `SPEAKER_00:` noise eating the context window.
* `export_srt` writes subtitles, handy for re-listening to a passage in a
  media player.
"""

import logging
from datetime import datetime
from pathlib import Path

from modules.audio_utils import format_timestamp

logger = logging.getLogger(__name__)

PARAGRAPH_GAP_SECONDS = 3.0
TIME_MARKER_INTERVAL = 300.0


def export(
    segments: list[dict],
    output_path: Path,
    include_timestamps: bool = True,
    speaker_names: dict[str, str] | None = None,
) -> Path:
    """Writes the full transcript with speaker labels.

    Format per line: `[HH:MM:SS] SPEAKER_00: text`, with a blank line
    between speaker changes.

    Args:
        segments: Dicts with "start", "end", "speaker", "text".
        output_path: Destination .txt path.
        include_timestamps: Whether to prefix each line with a timestamp.
        speaker_names: Optional map of diarization labels to real names,
            e.g. `{"SPEAKER_01": "Profesor"}`.

    Returns:
        The path to the saved file.

    Raises:
        ValueError: If segments is empty.
    """
    if not segments:
        raise ValueError("Segments list is empty. Nothing to export.")

    names = speaker_names or {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"# ClassTranscriber — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        previous_speaker = None
        for segment in segments:
            speaker = names.get(segment["speaker"], segment["speaker"])
            if previous_speaker is not None and speaker != previous_speaker:
                handle.write("\n")

            if include_timestamps:
                handle.write(
                    f"[{format_timestamp(segment['start'])}] {speaker}: {segment['text']}\n"
                )
            else:
                handle.write(f"{speaker}: {segment['text']}\n")
            previous_speaker = speaker

    logger.info("Transcript exported to %s (%d segments)", output_path, len(segments))
    return output_path


def export_professor(
    segments: list[dict],
    professor: str,
    output_path: Path,
    include_time_markers: bool = True,
    marker_interval: float = TIME_MARKER_INTERVAL,
    paragraph_gap: float = PARAGRAPH_GAP_SECONDS,
) -> Path:
    """Writes only the professor's speech, as continuous prose.

    Consecutive turns are joined into paragraphs, broken where the
    professor paused for longer than `paragraph_gap` — which usually maps
    to a change of topic or a student question in between. Sparse
    `[HH:MM]` markers let you find the passage in the recording without
    cluttering every line.

    Args:
        segments: Speaker-labelled segments.
        professor: The speaker label to keep.
        output_path: Destination .txt path.
        include_time_markers: Whether to emit periodic time markers.
        marker_interval: Minimum seconds between markers.
        paragraph_gap: Silence, in seconds, that starts a new paragraph.

    Returns:
        The path to the saved file.

    Raises:
        ValueError: If the professor has no segments.
    """
    speech = [
        segment
        for segment in segments
        if segment.get("speaker") == professor and segment.get("text", "").strip()
    ]
    if not speech:
        raise ValueError(f"No segments found for speaker '{professor}'.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    paragraphs: list[tuple[float, list[str]]] = []
    previous_end: float | None = None
    for segment in speech:
        if previous_end is None or segment["start"] - previous_end > paragraph_gap:
            paragraphs.append((segment["start"], []))
        paragraphs[-1][1].append(segment["text"].strip())
        previous_end = segment["end"]

    total_seconds = speech[-1]["end"] - speech[0]["start"]
    word_count = sum(len(segment["text"].split()) for segment in speech)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {output_path.stem} — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        handle.write(
            f"# Professor only ({professor}) — "
            f"{format_timestamp(total_seconds)} of speech, {word_count} words\n\n"
        )

        last_marker = -marker_interval
        for start, parts in paragraphs:
            if include_time_markers and start - last_marker >= marker_interval:
                handle.write(f"[{format_timestamp(start)}]\n")
                last_marker = start
            handle.write(" ".join(parts).strip() + "\n\n")

    logger.info(
        "Professor-only transcript exported to %s (%d words, %d paragraphs)",
        output_path,
        word_count,
        len(paragraphs),
    )
    return output_path


def export_markdown(
    segments: list[dict],
    output_path: Path,
    title: str,
    professor: str | None = None,
) -> Path:
    """Writes a Markdown transcript with the professor's turns highlighted.

    Args:
        segments: Speaker-labelled segments.
        output_path: Destination .md path.
        title: Document title.
        professor: Label rendered as "Profesor" instead of its raw id.

    Returns:
        The path to the saved file.
    """
    if not segments:
        raise ValueError("Segments list is empty. Nothing to export.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n")

        previous_speaker = None
        for segment in segments:
            speaker = segment["speaker"]
            if speaker != previous_speaker:
                display = "Profesor" if professor and speaker == professor else speaker
                handle.write(f"\n**{display}** _[{format_timestamp(segment['start'])}]_\n\n")
                previous_speaker = speaker
            handle.write(segment["text"].strip() + " ")

        handle.write("\n")

    logger.info("Markdown transcript exported to %s", output_path)
    return output_path


def export_srt(segments: list[dict], output_path: Path) -> Path:
    """Writes the transcript as an SRT subtitle file."""
    if not segments:
        raise ValueError("Segments list is empty. Nothing to export.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        for index, segment in enumerate(segments, 1):
            start = format_timestamp(segment["start"], with_millis=True)
            end = format_timestamp(segment["end"], with_millis=True)
            handle.write(f"{index}\n{start} --> {end}\n")
            handle.write(f"{segment['speaker']}: {segment['text'].strip()}\n\n")

    logger.info("Subtitles exported to %s (%d cues)", output_path, len(segments))
    return output_path
