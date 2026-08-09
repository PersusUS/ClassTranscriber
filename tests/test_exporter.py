"""Unit tests for M7 — modules/exporter.py."""

import re
from pathlib import Path

import pytest

from modules.exporter import export, export_markdown, export_professor, export_srt


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
