"""State and pure logic behind the desktop window.

Deliberately free of any Tkinter import: everything here is plain data and
functions, so it can be unit-tested without a display — and so the window
itself stays thin enough to read.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import config
from modules import pipeline

logger = logging.getLogger(__name__)

# Labels shown next to each stage in the window, in run order.
STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("record", "Grabar"),
    ("preprocess", "Limpiar ruido"),
    ("transcribe", "Transcribir"),
    ("diarize", "Separar voces"),
    ("merge", "Combinar"),
    ("clean", "Corregir con IA"),
    ("export", "Guardar transcripción"),
)

STATUS_MARKS = {
    "pending": "·",
    "running": "▶",
    "done": "✓",
    "skipped": "—",
    "failed": "✗",
}

# The level meter spans this range; below the floor there is nothing to see.
METER_FLOOR_DBFS = -60.0
METER_CEILING_DBFS = 0.0

# Matches the thresholds the recorder warns at, so the meter and the log agree.
QUIET_PEAK_DBFS = -45.0
CLIPPING_PEAK_DBFS = -1.0


@dataclass
class FormState:
    """Everything the user can choose in the window."""

    name: str = ""
    device: int | None = None           # Microphone index; None = system default
    profile: str = "auto"
    duration_minutes: int | None = None  # None = record until stopped
    num_speakers: int | None = None
    denoise: bool = True
    diarize: bool = True
    clean: bool = True
    input_path: Path | None = None      # Set when processing an existing file
    formats: tuple[str, ...] = ("txt", "professor", "md")
    extra: dict = field(default_factory=dict)


def default_session_name(now: datetime | None = None) -> str:
    """Builds a session name that sorts chronologically and is filename-safe.

    Pre-filling this is what makes the window usable thirty seconds before
    class starts: there is nothing to type.
    """
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M")
    return f"clase_{stamp}"


def sanitise_name(raw: str, now: datetime | None = None) -> str:
    """Turns whatever the user typed into a safe filename stem.

    Spaces become underscores and path separators are dropped, so a name
    like "Álgebra 2/3" cannot escape the output directory or break the
    session folder.
    """
    cleaned = (raw or "").strip().replace(" ", "_")
    cleaned = "".join(
        character for character in cleaned
        if character.isalnum() or character in "._-áéíóúüñÁÉÍÓÚÜÑ"
    ).strip("._-")
    return cleaned or default_session_name(now)


def parse_duration_minutes(text: str) -> int | None:
    """Reads the duration box, which is in minutes and may be left empty.

    Args:
        text: Raw contents of the entry box.

    Returns:
        Minutes as an int, or None for "record until I press Stop".

    Raises:
        ValueError: If the text is neither empty nor a positive number.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        minutes = int(float(stripped.replace(",", ".")))
    except ValueError:
        raise ValueError(f"«{text}» no es una duración válida. Usa minutos, por ejemplo 90.")
    if minutes <= 0:
        raise ValueError("La duración debe ser mayor que cero minutos.")
    return minutes


def parse_speakers(text: str) -> int | None:
    """Reads the optional speaker count.

    Raises:
        ValueError: If the text is neither empty nor a count of 1 or more.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        count = int(stripped)
    except ValueError:
        raise ValueError(f"«{text}» no es un número de personas válido.")
    if count < 1:
        raise ValueError("Tiene que hablar al menos una persona.")
    return count


def build_options(
    form: FormState,
    hf_token: str | None = None,
    **hooks,
) -> pipeline.PipelineOptions:
    """Translates the form into the options the pipeline expects.

    Args:
        form: The window's current state.
        hf_token: HuggingFace token, needed only for speaker separation.
        **hooks: `stop_event`, `on_level` and `on_stage` passed straight
            through to the pipeline.

    Returns:
        Ready-to-run `PipelineOptions`. The run is always non-interactive:
        a GUI must never block on a terminal prompt nobody can see.
    """
    settings = config.resolve_settings(profile=form.profile)

    return pipeline.PipelineOptions(
        name=sanitise_name(form.name),
        settings=settings,
        input_path=form.input_path,
        duration=form.duration_minutes * 60 if form.duration_minutes else None,
        device=form.device,
        professor="auto",
        num_speakers=form.num_speakers,
        denoise=form.denoise,
        diarize_enabled=form.diarize,
        clean_enabled=form.clean,
        interactive=False,
        hf_token=hf_token,
        formats=form.formats,
        **hooks,
    )


def missing_requirements(form: FormState, hf_token: str | None, ollama_ok: bool) -> list[str]:
    """Lists what the chosen options need but the machine does not have.

    Reported before the run starts, because discovering that Ollama is
    down *after* recording a two-hour lecture is a bad way to find out.
    """
    problems = []

    if form.diarize and not (hf_token or "").startswith("hf_"):
        problems.append(
            "Separar voces necesita un token de HuggingFace en el fichero .env. "
            "Desmarca «Separar voces del profesor y los alumnos» para transcribir "
            "sin distinguir quién habla."
        )

    if form.clean and not ollama_ok:
        problems.append(
            "Corregir con IA necesita Ollama en marcha (abre un terminal y ejecuta "
            "«ollama serve»). Desmarca «Corregir el texto con IA» para saltarte este paso."
        )

    return problems


def meter_fraction(dbfs: float) -> float:
    """Maps a dBFS level onto 0.0–1.0 for the level bar."""
    if dbfs <= METER_FLOOR_DBFS:
        return 0.0
    if dbfs >= METER_CEILING_DBFS:
        return 1.0
    return (dbfs - METER_FLOOR_DBFS) / (METER_CEILING_DBFS - METER_FLOOR_DBFS)


def meter_verdict(peak_dbfs: float) -> tuple[str, str]:
    """Judges the input level and says what to do about it.

    Returns:
        A `(state, message)` pair where state is "quiet", "clipping" or
        "ok". This is the single most useful thing the window shows: it
        tells you the microphone is not reaching the professor *before*
        the class, not after.
    """
    if peak_dbfs < QUIET_PEAK_DBFS:
        return "quiet", "Se oye muy bajo — acerca el micrófono al profesor"
    if peak_dbfs > CLIPPING_PEAK_DBFS:
        return "clipping", "Se satura — baja el volumen de entrada"
    return "ok", "Nivel correcto"


def format_elapsed(seconds: float) -> str:
    """Formats elapsed recording time as MM:SS or H:MM:SS."""
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def initial_stages(form: FormState) -> dict[str, str]:
    """Builds the stage checklist, pre-marking what this run will skip."""
    stages = {key: "pending" for key, _ in STAGE_LABELS}

    if form.input_path is not None:
        stages["record"] = "skipped"
    if not form.diarize:
        stages["diarize"] = "skipped"
    if not form.clean:
        stages["clean"] = "skipped"

    return stages


def describe_device(device: dict) -> str:
    """Renders a device dict from `list_input_devices()` for the dropdown."""
    suffix = " (predeterminado)" if device.get("default") else ""
    return f"{device['index']}: {device['name']}{suffix}"


def device_index_from_label(label: str) -> int | None:
    """Recovers the device index from a dropdown label."""
    head = (label or "").split(":", 1)[0].strip()
    return int(head) if head.isdigit() else None
