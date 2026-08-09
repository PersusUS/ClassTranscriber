"""M6 — LLM transcript cleanup via Ollama.

Whisper output is grammatically raw: no capitalisation of proper nouns,
missing accents, filler words, and the odd mangled technical term. A
small local model fixes that without anything leaving the machine.

The delicate part is *alignment*. The original implementation sent N lines
and matched the reply's lines back by position, so the moment the model
merged or split a line every following segment inherited the wrong
timestamp and, worse, the wrong speaker. Here each line is numbered and
matched back by its number; anything unmatched keeps its original text.
"""

import logging
import re

import config

logger = logging.getLogger(__name__)

# Bound lazily by `_get_chat()`; tests patch this attribute directly.
chat = None

_SYSTEM_PROMPTS = {
    "es": (
        "Eres un corrector de transcripciones de clases universitarias en español.\n"
        "Tarea: corrige ortografía, tildes y puntuación; añade mayúsculas donde "
        "corresponda; elimina muletillas (eh, em, o sea, ¿vale?, ¿no?) y "
        "repeticiones involuntarias; corrige errores evidentes de "
        "reconocimiento de voz usando el contexto.\n"
        "Reglas estrictas:\n"
        "- Conserva EXACTAMENTE la terminología técnica, fórmulas, siglas, "
        "nombres propios y cifras.\n"
        "- No resumas, no expliques, no añadas ni quites información.\n"
        "- No traduzcas: mantén el español.\n"
        "- Devuelve UNA línea por cada línea de entrada, con el mismo número "
        "y el formato 'N| texto'.\n"
        "- Si una línea es ininteligible, devuélvela tal cual."
    ),
    "en": (
        "You are a transcript corrector for university lectures in English.\n"
        "Task: fix spelling and punctuation, capitalise correctly, remove "
        "filler words (uh, um, like, you know) and accidental repetitions, "
        "and fix obvious speech-recognition errors from context.\n"
        "Strict rules:\n"
        "- Preserve technical terminology, formulas, acronyms, proper nouns "
        "and numbers EXACTLY.\n"
        "- Do not summarise, explain, add or remove information.\n"
        "- Return ONE line per input line, with the same number, formatted "
        "as 'N| text'.\n"
        "- If a line is unintelligible, return it unchanged."
    ),
}

_LINE_PATTERN = re.compile(r"^\s*(\d+)\s*[|.)\]:\-]\s*(.*)$")

# Set by `set_professor_speaker()`; only this speaker's text gets cleaned.
_professor_speaker: str | None = None


def set_professor_speaker(speaker: str | None) -> None:
    """Sets the professor's speaker label globally.

    Kept for backwards compatibility — `clean_transcript(professor=...)`
    is the clearer way to do the same thing.
    """
    global _professor_speaker
    _professor_speaker = speaker


def _get_chat():
    """Returns an Ollama chat callable bound to the configured host."""
    global chat
    if chat is None:
        from ollama import Client

        chat = Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT).chat
    return chat


def _build_chunks(
    indices: list[int],
    segments: list[dict],
    chunk_size: int,
    chunk_chars: int,
) -> list[list[int]]:
    """Splits segment indices into chunks bounded by count and characters.

    A fixed segment count is a poor bound: forty two-word interjections
    and forty long explanations differ by an order of magnitude in tokens,
    and overflowing the context window is what makes a small model start
    dropping lines.
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    current_chars = 0

    for index in indices:
        length = len(segments[index].get("text", "")) + 6   # numbering overhead
        if current and (len(current) >= chunk_size or current_chars + length > chunk_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(index)
        current_chars += length

    if current:
        chunks.append(current)
    return chunks


def _parse_reply(content: str, expected: int) -> dict[int, str]:
    """Parses a 'N| text' reply into a mapping of line number to text."""
    parsed: dict[int, str] = {}
    for line in content.splitlines():
        match = _LINE_PATTERN.match(line)
        if not match:
            continue
        number = int(match.group(1))
        text = match.group(2).strip()
        if 1 <= number <= expected and text:
            parsed[number] = text
    return parsed


def _clean_chunk(
    chunk_indices: list[int],
    segments: list[dict],
    model: str,
    language: str,
) -> int:
    """Cleans one chunk in place. Returns the number of lines replaced."""
    originals = [segments[index].get("text", "").strip() for index in chunk_indices]
    numbered = "\n".join(f"{number}| {text}" for number, text in enumerate(originals, 1))
    system_prompt = _SYSTEM_PROMPTS.get(language, _SYSTEM_PROMPTS["en"])

    try:
        response = _get_chat()(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": numbered},
            ],
            options={
                "temperature": config.OLLAMA_TEMPERATURE,
                "num_ctx": config.OLLAMA_NUM_CTX,
            },
        )
    except Exception as exc:      # noqa: BLE001 - normalised into one message below
        if _is_connection_error(exc):
            raise RuntimeError(
                f"Ollama is not reachable at {config.OLLAMA_HOST}. "
                "Start it with: ollama serve"
            ) from exc
        logger.warning("LLM call failed (%s) — keeping the original text", exc)
        return 0

    content = _extract_content(response)
    parsed = _parse_reply(content, len(originals))

    replaced = 0
    for position, index in enumerate(chunk_indices, 1):
        cleaned = parsed.get(position)
        if not cleaned:
            continue
        original = originals[position - 1]
        # A wildly longer line means the model started explaining or
        # hallucinating rather than correcting. Keep the original.
        if len(cleaned) > 3 * len(original) + 40:
            logger.debug("Discarding suspiciously long correction for line %d", position)
            continue
        segments[index]["text"] = cleaned
        segments[index]["original_text"] = original
        replaced += 1

    return replaced


def _extract_content(response) -> str:
    """Reads the assistant text out of an Ollama response of any shape."""
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", ""))
    return ""


def _is_connection_error(exc: Exception) -> bool:
    """Heuristically detects "Ollama is not running" style failures."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("connection", "refused", "unreachable", "failed to connect")
    )


def clean_transcript(
    segments: list[dict],
    model: str | None = None,
    professor: str | None = None,
    chunk_size: int = config.OLLAMA_CHUNK_SIZE,
    chunk_chars: int = config.OLLAMA_CHUNK_CHARS,
    language: str | None = None,
) -> list[dict]:
    """Cleans segment text with a local LLM.

    Args:
        segments: Dicts with "start", "end", "speaker", "text".
        model: Ollama model name. Defaults to the active profile's model.
        professor: Only clean this speaker's segments. Falls back to the
            value set via `set_professor_speaker()`; when neither is set,
            every segment is cleaned.
        chunk_size: Maximum segments per LLM call.
        chunk_chars: Maximum characters per LLM call.
        language: ISO 639-1 code selecting the prompt. Defaults to
            `config.LANGUAGE`.

    Returns:
        The same list, with cleaned "text" fields. Every modified segment
        keeps its previous wording under "original_text".

    Raises:
        RuntimeError: If Ollama is unreachable.
    """
    if not segments:
        logger.info("No segments to clean. Returning empty list.")
        return segments

    model = model or config.PROFILES[config.resolve_profile().name].ollama_model
    language = (language or config.LANGUAGE).lower()
    target = professor if professor is not None else _professor_speaker

    if target:
        indices = [
            index for index, segment in enumerate(segments) if segment.get("speaker") == target
        ]
        if not indices:
            logger.warning("No segments found for speaker '%s' — nothing cleaned", target)
            return segments
        logger.info("Cleaning %d segment(s) from %s only", len(indices), target)
    else:
        indices = list(range(len(segments)))
        logger.info("Cleaning all %d segment(s)", len(indices))

    chunks = _build_chunks(indices, segments, chunk_size, chunk_chars)
    logger.info("Cleaning in %d chunk(s) with model %s", len(chunks), model)

    total_replaced = 0
    for number, chunk_indices in enumerate(chunks, 1):
        replaced = _clean_chunk(chunk_indices, segments, model, language)
        total_replaced += replaced

        if replaced < len(chunk_indices) * 0.6:
            logger.warning(
                "Chunk %d/%d: only %d of %d lines were returned correctly; "
                "the rest keep their original text",
                number,
                len(chunks),
                replaced,
                len(chunk_indices),
            )
        else:
            logger.info("Chunk %d/%d cleaned (%d lines)", number, len(chunks), replaced)

    logger.info("Cleanup complete: %d of %d segments rewritten", total_replaced, len(indices))
    return segments
