"""Unit tests for M7 — modules/exporter.py."""

import json
import re
from pathlib import Path

import pytest

from modules.exporter import (
    VALID_FORMATS,
    export,
    export_markdown,
    export_professor,
    export_srt,
    parse_formats,
    write_outputs,
)


def _make_segments():
    """Sample segments with two speakers."""
    return [
        {"start": 2.0, "end": 10.0, "speaker": "SPEAKER_00",
         "text": "Bienvenidos, hoy hablamos de computación distribuida."},
        {"start": 15.0, "end": 22.0, "speaker": "SPEAKER_01",
         "text": "Profesor, ¿puede explicar la consistencia?"},
        {"start": 22.5, "end": 65.0, "speaker": "SPEAKER_00",
         "text": "Sí, la consistencia es que todos los nodos vean el mismo dato."},
    ]


def test_export_creates_file(tmp_path: Path):
    """Assert the output file is created."""
    output = tmp_path / "transcript.txt"
    export(_make_segments(), output)
    assert output.exists()


def test_export_contains_speaker_labels(tmp_path: Path):
    """Assert the transcript carries speaker labels."""
    output = tmp_path / "transcript.txt"
    export(_make_segments(), output)
    assert "SPEAKER_" in output.read_text(encoding="utf-8")


def test_export_timestamp_format(tmp_path: Path):
    """Assert timestamps match [HH:MM:SS]."""
    output = tmp_path / "transcript.txt"
    export(_make_segments(), output)
    assert re.findall(r"\[\d{2}:\d{2}:\d{2}\]", output.read_text(encoding="utf-8"))


def test_export_empty_segments(tmp_path: Path):
    """Assert ValueError for empty input."""
    with pytest.raises(ValueError):
        export([], tmp_path / "transcript.txt")


def test_export_utf8(tmp_path: Path):
    """Accents and other non-ASCII characters survive the round trip."""
    segments = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00",
                 "text": "Héllo wörld — señor André 你好"}]
    output = tmp_path / "transcript.txt"
    export(segments, output)
    assert "Héllo wörld — señor André 你好" in output.read_text(encoding="utf-8")


def test_export_renames_speakers(tmp_path: Path):
    """A label map replaces the raw diarization ids."""
    output = tmp_path / "transcript.txt"
    export(_make_segments(), output, speaker_names={"SPEAKER_00": "PROFESOR"})
    content = output.read_text(encoding="utf-8")
    assert "PROFESOR:" in content
    assert "SPEAKER_00" not in content


def test_export_professor_excludes_other_speakers(tmp_path: Path):
    """The professor-only file must contain nothing a student said."""
    output = tmp_path / "profesor.txt"
    export_professor(_make_segments(), "SPEAKER_00", output)
    content = output.read_text(encoding="utf-8")

    assert "computación distribuida" in content
    assert "¿puede explicar la consistencia?" not in content
    assert "SPEAKER_" not in content.split("\n\n", 1)[1]     # No labels in the body


def test_export_professor_groups_paragraphs(tmp_path: Path):
    """Turns close in time join into one paragraph; a long pause splits them."""
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "P", "text": "Primera frase."},
        {"start": 2.5, "end": 4.0, "speaker": "P", "text": "Segunda frase."},
        {"start": 60.0, "end": 62.0, "speaker": "P", "text": "Tema nuevo."},
    ]
    output = tmp_path / "profesor.txt"
    export_professor(segments, "P", output)

    body = output.read_text(encoding="utf-8").split("\n\n", 1)[1]
    paragraphs = [line for line in body.split("\n\n") if line.strip()]

    assert "Primera frase. Segunda frase." in paragraphs[0]
    assert any("Tema nuevo." in paragraph for paragraph in paragraphs)


def test_export_professor_writes_time_markers(tmp_path: Path):
    """Sparse [HH:MM:SS] markers let you find the passage in the recording."""
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "P", "text": "Inicio."},
        {"start": 900.0, "end": 902.0, "speaker": "P", "text": "Quince minutos después."},
    ]
    output = tmp_path / "profesor.txt"
    export_professor(segments, "P", output)

    assert "[00:15:00]" in output.read_text(encoding="utf-8")


def test_export_professor_unknown_speaker(tmp_path: Path):
    """Assert ValueError when the requested speaker never spoke."""
    with pytest.raises(ValueError):
        export_professor(_make_segments(), "SPEAKER_99", tmp_path / "profesor.txt")


def test_export_markdown(tmp_path: Path):
    """The professor's label is rendered by name in Markdown."""
    output = tmp_path / "clase.md"
    export_markdown(_make_segments(), output, title="Clase 1", professor="SPEAKER_00")
    content = output.read_text(encoding="utf-8")

    assert content.startswith("# Clase 1")
    assert "**Profesor**" in content


def test_export_srt_format(tmp_path: Path):
    """SRT cues use comma-separated milliseconds."""
    output = tmp_path / "clase.srt"
    export_srt(_make_segments(), output)
    content = output.read_text(encoding="utf-8")

    assert content.startswith("1\n")
    assert re.search(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", content)


def test_parse_formats_normalises():
    """Case and spacing are tolerated; duplicates collapse."""
    assert parse_formats(" TXT , professor,txt ") == ("txt", "professor")


def test_parse_formats_rejects_typos():
    """The Spanish spelling of 'professor' is the obvious typo to catch."""
    with pytest.raises(ValueError, match="profesor"):
        parse_formats("txt,profesor")


def test_parse_formats_rejects_empty():
    """An empty format list is a mistake, not "write nothing"."""
    with pytest.raises(ValueError, match="No output format"):
        parse_formats("  ,  ")


def test_parse_formats_accepts_every_valid_name():
    """Every advertised format is actually accepted."""
    assert parse_formats(",".join(VALID_FORMATS)) == VALID_FORMATS


def test_write_outputs_writes_each_format(tmp_path: Path):
    """One call produces every requested artifact."""
    outputs = write_outputs(
        _make_segments(),
        name="clase",
        output_dir=tmp_path / "out",
        formats=("txt", "professor", "md", "srt", "json"),
        professor="SPEAKER_00",
    )

    assert set(outputs) == {"full", "professor", "markdown", "subtitles", "json"}
    assert all(path.exists() for path in outputs.values())


def test_write_outputs_creates_the_directory(tmp_path: Path):
    """The output directory does not have to exist beforehand."""
    write_outputs(_make_segments(), "clase", tmp_path / "nueva", formats=("txt",))

    assert (tmp_path / "nueva" / "clase_completo.txt").exists()


def test_write_outputs_skips_professor_file_without_a_professor(tmp_path: Path, caplog):
    """Asking for professor-only output with no professor warns and skips."""
    with caplog.at_level("WARNING"):
        outputs = write_outputs(
            _make_segments(), "clase", tmp_path, formats=("professor",), professor=None
        )

    assert "professor" not in outputs
    assert "no professor was identified" in caplog.text


def test_write_outputs_rejects_unknown_format(tmp_path: Path):
    """An unknown format is rejected here too, not only at the CLI."""
    with pytest.raises(ValueError, match="Unknown output format"):
        write_outputs(_make_segments(), "clase", tmp_path, formats=("pdf",))


def test_write_outputs_json_is_readable(tmp_path: Path):
    """The JSON export round-trips, accents included."""
    outputs = write_outputs(_make_segments(), "clase", tmp_path, formats=("json",))
    data = json.loads(outputs["json"].read_text(encoding="utf-8"))

    assert data[1]["text"].startswith("Profesor, ¿puede")


def test_export_professor_counts_only_the_professor_speech(tmp_path: Path):
    """The header's speech time excludes the gaps where students talked.

    Measuring first-to-last would count the student's question as the
    professor's speaking time.
    """
    segments = [
        {"start": 0.0, "end": 10.0, "speaker": "P", "text": "Primera."},
        {"start": 100.0, "end": 110.0, "speaker": "P", "text": "Segunda."},
    ]
    output = tmp_path / "profesor.txt"
    export_professor(segments, "P", output)

    assert "00:00:20 de habla" in output.read_text(encoding="utf-8")


def test_export_markdown_rejects_empty(tmp_path: Path):
    """An empty transcript is an error, not an empty document."""
    with pytest.raises(ValueError):
        export_markdown([], tmp_path / "a.md", title="x")


def test_export_srt_rejects_empty(tmp_path: Path):
    """Same for subtitles."""
    with pytest.raises(ValueError):
        export_srt([], tmp_path / "a.srt")


def test_export_without_timestamps(tmp_path: Path):
    """Timestamps can be omitted for a cleaner reading copy."""
    output = tmp_path / "plain.txt"
    export(_make_segments(), output, include_timestamps=False)
    content = output.read_text(encoding="utf-8")

    assert "SPEAKER_00:" in content
    assert not re.search(r"\[\d{2}:\d{2}:\d{2}\] SPEAKER", content)


def test_export_professor_without_time_markers(tmp_path: Path):
    """Time markers can be turned off entirely."""
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "P", "text": "Inicio."},
        {"start": 900.0, "end": 902.0, "speaker": "P", "text": "Después."},
    ]
    output = tmp_path / "profesor.txt"
    export_professor(segments, "P", output, include_time_markers=False)

    body = output.read_text(encoding="utf-8").split("\n\n", 1)[1]
    assert "[00:15:00]" not in body


def test_export_headers_are_in_spanish(tmp_path: Path):
    """The files the user actually reads are written in Spanish.

    The transcript body is Spanish, so an English header in the same file
    is just noise — including for the LLM that gets fed the professor file.
    """
    professor_file = tmp_path / "clase_profesor.txt"
    full_file = tmp_path / "clase_completo.txt"
    export_professor(_make_segments(), "SPEAKER_00", professor_file)
    export(_make_segments(), full_file)

    professor_header = professor_file.read_text(encoding="utf-8").splitlines()[1]
    full_header = full_file.read_text(encoding="utf-8").splitlines()[0]

    assert "Solo el profesor" in professor_header
    assert "de habla" in professor_header
    assert "palabras" in professor_header
    assert "transcripción completa" in full_header


def test_export_dates_use_day_first(tmp_path: Path):
    """Dates read as DD/MM/YYYY, which is what a Spanish reader expects."""
    output = tmp_path / "clase.txt"
    export(_make_segments(), output)

    assert re.search(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", output.read_text(encoding="utf-8"))
