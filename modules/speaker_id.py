"""Professor identification.

Diarization returns anonymous labels (SPEAKER_00, SPEAKER_01…) and their
numbering depends on who happens to talk first, so it cannot be hardcoded.

In a lecture the professor is whoever talks the most — usually by a wide
margin. That is a reliable enough default to run unattended, and when the
margin is *not* wide (a seminar, a lab session, a heavy Q&A) the caller
can confirm interactively with a sample of each speaker's words.
"""

import logging
import sys

import config
from modules.audio_utils import format_timestamp

logger = logging.getLogger(__name__)


def speaker_stats(segments: list[dict]) -> list[dict]:
    """Summarises how much each speaker talked.

    Args:
        segments: Speaker-labelled segments with "start", "end", "text".

    Returns:
        A list of dicts with keys "speaker", "seconds", "share", "turns"
        and "words", sorted by speaking time descending.
    """
    totals: dict[str, dict] = {}
    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        entry = totals.setdefault(
            speaker, {"speaker": speaker, "seconds": 0.0, "turns": 0, "words": 0}
        )
        entry["seconds"] += max(0.0, segment["end"] - segment["start"])
        entry["turns"] += 1
        entry["words"] += len(segment.get("text", "").split())

    total_seconds = sum(entry["seconds"] for entry in totals.values()) or 1.0
    stats = sorted(totals.values(), key=lambda entry: entry["seconds"], reverse=True)
    for entry in stats:
        entry["share"] = entry["seconds"] / total_seconds
    return stats


def format_stats(stats: list[dict]) -> str:
    """Renders speaker statistics as an aligned text table."""
    lines = [f"{'#':<3} {'SPEAKER':<14} {'TIME':>10} {'SHARE':>7} {'TURNS':>7} {'WORDS':>7}"]
    for index, entry in enumerate(stats):
        lines.append(
            f"{index:<3} {entry['speaker']:<14} "
            f"{format_timestamp(entry['seconds']):>10} "
            f"{entry['share'] * 100:>6.1f}% {entry['turns']:>7} {entry['words']:>7}"
        )
    return "\n".join(lines)


def sample_lines(segments: list[dict], speaker: str, count: int = 3) -> list[str]:
    """Returns the speaker's longest utterances, for identification by ear.

    The longest turns are the most recognisable: "¿puede repetir la
    última parte?" reads very differently from a lecture explanation.
    """
    candidates = [
        segment
        for segment in segments
        if segment.get("speaker") == speaker and segment.get("text", "").strip()
    ]
    candidates.sort(key=lambda segment: len(segment["text"]), reverse=True)

    lines = []
    for segment in candidates[:count]:
        text = segment["text"].strip()
        if len(text) > 160:
            text = text[:157] + "..."
        lines.append(f"[{format_timestamp(segment['start'])}] {text}")
    return lines


def resolve_professor(
    segments: list[dict],
    requested: str | None = None,
    interactive: bool = True,
    confidence_share: float = config.PROFESSOR_CONFIDENCE_SHARE,
) -> str | None:
    """Decides which diarization label belongs to the professor.

    Args:
        segments: Merged, speaker-labelled segments.
        requested: An explicit choice. Accepts a full label
            ("SPEAKER_01"), a rank index ("0" for the most talkative),
            "auto" to always take the most talkative speaker, or "none"
            to skip professor filtering. None means "auto, but ask if the
            result looks uncertain".
        interactive: Whether asking the user is allowed. Automatically
            disabled when stdin is not a terminal.
        confidence_share: Share of total speaking time above which the top
            speaker is accepted without asking.

    Returns:
        The chosen speaker label, or None when no professor should be
        singled out.
    """
    stats = speaker_stats(segments)
    if not stats:
        return None

    logger.info("Speaker breakdown:\n%s", format_stats(stats))

    labels = [entry["speaker"] for entry in stats]

    if requested:
        choice = requested.strip()
        if choice.lower() in {"none", "off", "all"}:
            logger.info("Professor filtering disabled by request")
            return None
        if choice.lower() != "auto":
            resolved = _match_label(choice, labels)
            if resolved is None:
                raise ValueError(
                    f"Unknown speaker '{requested}'. Available: {', '.join(labels)}"
                )
            logger.info("Professor set explicitly: %s", resolved)
            return resolved

    top = stats[0]
    confident = top["share"] >= confidence_share or len(stats) == 1
    explicit_auto = bool(requested) and requested.strip().lower() == "auto"

    if confident or explicit_auto or not (interactive and sys.stdin.isatty()):
        if not confident and not explicit_auto:
            logger.warning(
                "Top speaker %s only accounts for %.0f%% of speaking time — "
                "verify the transcript, or re-run with --professor to override.",
                top["speaker"],
                top["share"] * 100,
            )
        logger.info(
            "Professor identified as %s (%.0f%% of speaking time)",
            top["speaker"],
            top["share"] * 100,
        )
        return top["speaker"]

    return _ask(segments, stats)


def _match_label(choice: str, labels: list[str]) -> str | None:
    """Resolves a user-supplied speaker reference to a real label."""
    if choice in labels:
        return choice

    upper = choice.upper()
    for label in labels:
        if label.upper() == upper:
            return label

    if choice.isdigit():
        rank = int(choice)
        if 0 <= rank < len(labels):
            return labels[rank]

    # Allow "1" to mean SPEAKER_01 as well as the second-most-talkative.
    for label in labels:
        if label.upper().endswith(upper.zfill(2)):
            return label

    return None


def _ask(segments: list[dict], stats: list[dict]) -> str | None:
    """Prompts the user to pick the professor, showing sample utterances."""
    print("\nWhich speaker is the professor?\n")
    print(format_stats(stats))
    print()
    for index, entry in enumerate(stats):
        print(f"[{index}] {entry['speaker']} — {entry['share'] * 100:.0f}% of the time")
        for line in sample_lines(segments, entry["speaker"]):
            print(f"      {line}")
        print()

    labels = [entry["speaker"] for entry in stats]
    while True:
        try:
            answer = input(
                f"Enter a number 0-{len(labels) - 1}, a speaker label, "
                f"or 'none' to keep every speaker [0]: "
            ).strip()
        except EOFError:
            return labels[0]

        if not answer:
            return labels[0]
        if answer.lower() in {"none", "off", "all"}:
            return None

        resolved = _match_label(answer, labels)
        if resolved:
            return resolved
        print(f"Not a valid choice. Available: {', '.join(labels)}")
