"""M6 — LLM text cleanup module.

Uses Ollama + gemma3:4b to clean transcription text: fix grammar,
punctuation, remove filler words, and make the text coherent.
"""

import logging
import math

from ollama import chat

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a transcript cleaner for academic lectures. The speaker has a strong Chinese accent and speaks non-native English.\n"
    "Your task: fix grammar, add correct punctuation, remove filler words (uh, um, er, like, you know), \n"
    "correct obvious transcription errors, and make the text read naturally.\n"
    "Preserve all technical and academic terminology exactly as-is.\n"
    "Respond ONLY with the corrected text, one line per input line. Do not add explanations or numbering."
)

_professor_speaker = None


def set_professor_speaker(speaker: str):
    """Sets the professor speaker ID. Only this speaker's text will be cleaned."""
    global _professor_speaker
    _professor_speaker = speaker


def clean_transcript(
    segments: list[dict],
    model: str = "gemma3:4b",
    chunk_size: int = 50,
) -> list[dict]:
    """Cleans the text of all segments using Ollama.

    Processes segments in chunks of chunk_size to stay within context limits.

    Args:
        segments: List of dicts with keys "start", "end", "speaker", "text".
        model: Ollama model name. Default "gemma3:4b".
        chunk_size: Number of segments per LLM call.

    Returns:
        The same list structure with cleaned "text" fields.

    Raises:
        RuntimeError: If Ollama is not running.
    """
    if not segments:
        logger.info("No segments to clean. Returning empty list.")
        return segments

    if getattr(clean_transcript, '__globals__', {}).get('_professor_speaker', None) or globals().get('_professor_speaker', None):
        all_indices = [
            i
            for i, s in enumerate(segments)
            if s.get("speaker") == _professor_speaker
        ]
    else:
        all_indices = list(range(len(segments)))

    if not all_indices:
        logger.info("No professor segments found. Returning unaltered.")
        return segments

    num_chunks = math.ceil(len(all_indices) / chunk_size)
    logger.info(
        "Cleaning %d segments in %d chunk(s) (model=%s)",
        len(all_indices),
        num_chunks,
        model,
    )

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_indices = all_indices[start:end]

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
