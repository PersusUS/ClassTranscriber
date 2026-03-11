"""Unit tests for M7 — modules/exporter.py."""

import re
from pathlib import Path

import pytest

from modules.exporter import export


def _make_segments():
    """Helper: returns sample segments with two speakers."""
    return [
        {"start": 2.0, "end": 10.0, "speaker": "SPEAKER_00", "text": "Welcome everyone, today we discuss distributed computing."},
        {"start": 15.0, "end": 22.0, "speaker": "SPEAKER_01", "text": "Professor, can you explain consistency?"},
        {"start": 22.5, "end": 65.0, "speaker": "SPEAKER_00", "text": "Yes, consistency refers to all nodes observing the same data."},
    ]


def test_export_creates_file(tmp_path: Path):
    """Assert output file is created."""
    output_path = tmp_path / "transcript.txt"
    export(_make_segments(), output_path)
    assert output_path.exists()


def test_export_contains_speaker_labels(tmp_path: Path):
    """Assert output file text contains 'SPEAKER_'."""
    output_path = tmp_path / "transcript.txt"
    export(_make_segments(), output_path)
    content = output_path.read_text(encoding="utf-8")
    assert "SPEAKER_" in content


def test_export_timestamp_format(tmp_path: Path):
    """Assert timestamps match pattern [HH:MM:SS]."""
    output_path = tmp_path / "transcript.txt"
    export(_make_segments(), output_path)
    content = output_path.read_text(encoding="utf-8")
    matches = re.findall(r"\[\d{2}:\d{2}:\d{2}\]", content)
    assert len(matches) >= 1


def test_export_empty_segments(tmp_path: Path):
    """Assert ValueError is raised for empty input."""
    output_path = tmp_path / "transcript.txt"
    with pytest.raises(ValueError):
        export([], output_path)


def test_export_utf8(tmp_path: Path):
    """Write a segment with special characters and assert they are preserved."""
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Héllo wörld — señor André 你好"},
    ]
    output_path = tmp_path / "transcript.txt"
    export(segments, output_path)
    content = output_path.read_text(encoding="utf-8")
    assert "Héllo wörld — señor André 你好" in content
