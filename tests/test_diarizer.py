"""Unit tests for M3 — modules/diarizer.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import modules.diarizer as diarizer_mod
from modules.diarizer import _resolve_device, diarize


class FakeSegment:
    """Mimics a pyannote Segment with start/end attributes."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class FakeDiarization:
    """Mimics the pyannote diarization result object."""

    def __init__(self, tracks: list[tuple]):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        return iter(self._tracks)


@pytest.fixture(autouse=True)
def clear_pipeline_cache():
    """Prevents the cached pipeline from leaking between tests."""
    diarizer_mod._pipelines.clear()
    yield
    diarizer_mod._pipelines.clear()


def _pipeline_returning(tracks):
    """Builds a mock pipeline yielding the given tracks."""
    pipeline = MagicMock()
    pipeline.return_value = FakeDiarization(tracks)
    return pipeline


@pytest.fixture
def fake_pipeline():
    """A pipeline that returns three segments from two speakers."""
    return _pipeline_returning(
        [
            (FakeSegment(0.0, 5.0), None, "SPEAKER_00"),
            (FakeSegment(5.5, 12.0), None, "SPEAKER_01"),
            (FakeSegment(12.5, 20.0), None, "SPEAKER_00"),
        ]
    )


@patch("modules.diarizer._load_pipeline")
def test_diarize_returns_list(mock_load, fake_pipeline, tmp_path):
    """Assert the return type is a list."""
    mock_load.return_value = fake_pipeline
    audio = tmp_path / "test.wav"
    audio.touch()

    assert isinstance(diarize(audio, hf_token="fake_token"), list)


@patch("modules.diarizer._load_pipeline")
def test_diarize_segment_keys(mock_load, fake_pipeline, tmp_path):
    """Assert each dict contains 'start', 'end' and 'speaker'."""
    mock_load.return_value = fake_pipeline
    audio = tmp_path / "test.wav"
    audio.touch()

    for segment in diarize(audio, hf_token="fake_token"):
        assert {"start", "end", "speaker"} <= set(segment)


@patch("modules.diarizer._load_pipeline")
def test_diarize_sorted_by_start(mock_load, tmp_path):
    """Assert segments come back sorted by start time."""
    mock_load.return_value = _pipeline_returning(
        [
            (FakeSegment(10.0, 15.0), None, "SPEAKER_01"),
            (FakeSegment(0.0, 5.0), None, "SPEAKER_00"),
            (FakeSegment(5.5, 9.0), None, "SPEAKER_00"),
        ]
    )
    audio = tmp_path / "test.wav"
    audio.touch()

    starts = [segment["start"] for segment in diarize(audio, hf_token="fake_token")]

    assert starts == sorted(starts)


@patch("modules.diarizer._load_pipeline")
def test_diarize_passes_exact_speaker_count(mock_load, fake_pipeline, tmp_path):
    """num_speakers must replace the min/max bounds when given."""
    mock_load.return_value = fake_pipeline
    audio = tmp_path / "test.wav"
    audio.touch()

    diarize(audio, hf_token="fake_token", num_speakers=3)

    kwargs = fake_pipeline.call_args.kwargs
    assert kwargs["num_speakers"] == 3
    assert "max_speakers" not in kwargs


def test_diarize_invalid_path():
    """Assert FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        diarize(Path("/nonexistent/audio.wav"), hf_token="fake_token")


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    """Requesting CUDA on a machine without it must not fail — CPU is fine."""
    monkeypatch.setattr("config.cuda_available", lambda: False)

    assert _resolve_device("cuda") == "cpu"
    assert _resolve_device("auto") == "cpu"
    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_prefers_cuda_when_present(monkeypatch):
    """With CUDA available, 'auto' selects the GPU."""
    monkeypatch.setattr("config.cuda_available", lambda: True)

    assert _resolve_device("auto") == "cuda"


def _fake_torch(monkeypatch):
    """Injects a fake torch module so _load_pipeline can be exercised."""
    torch = MagicMock()
    torch.device.side_effect = lambda name: f"device:{name}"
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _fake_pyannote(monkeypatch, pipeline_or_error):
    """Injects a fake pyannote.audio whose from_pretrained is controllable."""
    module = MagicMock()
    if isinstance(pipeline_or_error, Exception):
        module.Pipeline.from_pretrained.side_effect = pipeline_or_error
    else:
        module.Pipeline.from_pretrained.return_value = pipeline_or_error
    monkeypatch.setitem(sys.modules, "pyannote.audio", module)
    return module


def test_load_pipeline_moves_to_device(monkeypatch, fake_pipeline):
    """The pipeline is moved onto the resolved device."""
    _fake_torch(monkeypatch)
    _fake_pyannote(monkeypatch, fake_pipeline)

    loaded = diarizer_mod._load_pipeline("model", "hf_token", "cpu")

    assert loaded is fake_pipeline
    fake_pipeline.to.assert_called_once_with("device:cpu")


def test_load_pipeline_is_cached(monkeypatch, fake_pipeline):
    """Loading twice reuses the weights instead of re-downloading."""
    _fake_torch(monkeypatch)
    module = _fake_pyannote(monkeypatch, fake_pipeline)

    diarizer_mod._load_pipeline("model", "hf_token", "cpu")
    diarizer_mod._load_pipeline("model", "hf_token", "cpu")

    assert module.Pipeline.from_pretrained.call_count == 1


def test_load_pipeline_batches_only_on_gpu(monkeypatch, fake_pipeline):
    """Bigger batches help on GPU but only inflate memory on CPU."""
    _fake_torch(monkeypatch)
    _fake_pyannote(monkeypatch, fake_pipeline)
    fake_pipeline.segmentation_batch_size = 1

    diarizer_mod._load_pipeline("model", "hf_token", "cuda")

    assert fake_pipeline.segmentation_batch_size == 32


def test_load_pipeline_explains_a_missing_licence(monkeypatch):
    """pyannote returns None when the licence was not accepted.

    That silent None is the single most common setup failure, so it has to
    turn into instructions rather than an AttributeError later on.
    """
    _fake_torch(monkeypatch)
    _fake_pyannote(monkeypatch, None)

    with pytest.raises(RuntimeError, match="huggingface.co"):
        diarizer_mod._load_pipeline("model", None, "cpu")


def test_load_pipeline_explains_a_download_failure(monkeypatch):
    """A raised error is wrapped with the same actionable guidance."""
    _fake_torch(monkeypatch)
    _fake_pyannote(monkeypatch, OSError("401 Unauthorized"))

    with pytest.raises(RuntimeError, match="settings/tokens"):
        diarizer_mod._load_pipeline("model", "hf_token", "cpu")


@patch("modules.diarizer._load_pipeline")
def test_diarize_reads_token_from_environment(mock_load, fake_pipeline, tmp_path, monkeypatch):
    """HF_TOKEN is picked up when no token is passed explicitly."""
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    mock_load.return_value = fake_pipeline
    audio = tmp_path / "test.wav"
    audio.touch()

    diarize(audio)

    assert mock_load.call_args.args[1] == "hf_from_env"


@patch("modules.diarizer._load_pipeline")
def test_diarize_defaults_to_speaker_bounds(mock_load, fake_pipeline, tmp_path):
    """Without an exact count, min/max bounds are passed instead."""
    mock_load.return_value = fake_pipeline
    audio = tmp_path / "test.wav"
    audio.touch()

    diarize(audio, hf_token="t", max_speakers=4)

    kwargs = fake_pipeline.call_args.kwargs
    assert kwargs["max_speakers"] == 4
    assert "num_speakers" not in kwargs


def test_null_hook_is_a_no_op_context():
    """The fallback hook can be entered and exited safely."""
    with diarizer_mod._NullHook() as hook:
        assert hook is None


def test_progress_hook_is_used_when_available(monkeypatch, fake_pipeline, tmp_path):
    """pyannote's progress bar is passed through when the class exists."""
    hook_instance = MagicMock()
    hook_instance.__enter__ = MagicMock(return_value="the-hook")
    hook_instance.__exit__ = MagicMock(return_value=False)

    hook_module = MagicMock()
    hook_module.ProgressHook.return_value = hook_instance
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines.utils.hook", hook_module)

    audio = tmp_path / "test.wav"
    audio.touch()

    with patch("modules.diarizer._load_pipeline", return_value=fake_pipeline):
        diarize(audio, hf_token="t")

    assert fake_pipeline.call_args.kwargs["hook"] == "the-hook"


def test_progress_hook_falls_back_when_absent(monkeypatch):
    """Without the hook class, a no-op context is used instead."""
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines.utils.hook", None)

    assert isinstance(diarizer_mod._progress_hook(), diarizer_mod._NullHook)
