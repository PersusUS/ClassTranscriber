"""M6 — LLM text cleanup module.

Uses Ollama + gemma3:4b to clean transcription text: fix grammar,
punctuation, remove filler words, and make the text coherent.
"""

import logging
import math

from ollama import chat

logger = logging.getLogger(__name__)

_professor_speaker: str | None = None

_SYSTEM_PROMPT = (
    "You are a transcript cleaner for academic lectures. The speaker has a strong Chinese accent and speaks non-native English.\n"
    "Your task: fix grammar, add correct punctuation, remove filler words (uh, um, er, like, you know), \n"
    "correct obvious transcription errors, and make the text read naturally.\n"
    "Preserve all technical and academic terminology exactly as-is.\n"
    "Respond ONLY with the corrected text, one line per input line. Do not add explanations or numbering."
)


def set_professor_speaker(speaker_label: str) -> None:
    """Sets the module-level variable that identifies which speaker is the professor.

    Must be called before clean_transcript().

    Args:
        speaker_label: The speaker label string (e.g. "SPEAKER_00").
    """
    global _professor_speaker
    _professor_speaker = speaker_label
    logger.info("Professor speaker set to: %s", speaker_label)


def clean_transcript(
    segments: list[dict],
    model: str = "gemma3:4b",
    chunk_size: int = 50,
) -> list[dict]:
    """Takes merged segments and cleans the professor's text using Ollama.

    Non-professor segments are passed through unchanged.
    Processes segments in chunks of chunk_size to stay within context limits.

    Args:
        segments: List of dicts with keys "start", "end", "speaker", "text".
        model: Ollama model name. Default "gemma3:4b".
        chunk_size: Number of professor segments per LLM call.

    Returns:
        The same list structure with cleaned "text" fields for professor segments.

    Raises:
        RuntimeError: If _professor_speaker has not been set.
        RuntimeError: If Ollama is not running.
    """
    if _professor_speaker is None:
        raise RuntimeError(
            "Professor speaker not set. Call set_professor_speaker() first."
        )

    # Collect indices of professor segments
    prof_indices = [
        i for i, seg in enumerate(segments)
        if seg["speaker"] == _professor_speaker
    ]

    if not prof_indices:
        logger.info("No professor segments found. Returning segments unchanged.")
        return segments

    num_chunks = math.ceil(len(prof_indices) / chunk_size)
    logger.info(
        "Cleaning %d professor segments in %d chunk(s) (model=%s)",
        len(prof_indices),
        num_chunks,
        model,
    )

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_indices = prof_indices[start:end]

        original_texts = [segments[i]["text"] for i in chunk_indices]
        user_message = "\n".join(original_texts)

        try:
            response = chat(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
        except ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve"
            ) from exc

        cleaned_lines = response.message.content.strip().split("\n")

        # Assign cleaned text back; pad with originals if LLM returned fewer lines
        for j, idx in enumerate(chunk_indices):
            if j < len(cleaned_lines) and cleaned_lines[j].strip():
                segments[idx]["text"] = cleaned_lines[j].strip()
            # else: keep original text

        logger.info("Chunk %d/%d cleaned", chunk_idx + 1, num_chunks)

    return segments
