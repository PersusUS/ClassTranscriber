"""Unit tests for config.py — profile and device resolution."""

import sys
from unittest.mock import MagicMock

import pytest

import config


def test_profiles_are_ordered_by_cost():
    """Each profile up the ladder uses a larger model or more search."""
    low = config.PROFILES["low"]
    balanced = config.PROFILES["balanced"]
    quality = config.PROFILES["quality"]

    assert low.whisper_model == "small"
    assert low.beam_size < quality.beam_size
    assert balanced.whisper_model != quality.whisper_model


def test_resolve_profile_auto_without_cuda(monkeypatch):
    """A laptop with no GPU gets the balanced profile, never 'quality'."""
    monkeypatch.setattr(config, "cuda_available", lambda: False)

    assert config.resolve_profile("auto").name == "balanced"


def test_resolve_profile_auto_with_big_gpu(monkeypatch):
    """A GPU with enough VRAM unlocks the quality profile."""
    monkeypatch.setattr(config, "cuda_available", lambda: True)
    monkeypatch.setattr(config, "gpu_vram_gb", lambda: 8.0)

    assert config.resolve_profile("auto").name == "quality"


def test_resolve_profile_auto_with_small_gpu(monkeypatch):
    """A small GPU stays on the balanced profile."""
    monkeypatch.setattr(config, "cuda_available", lambda: True)
    monkeypatch.setattr(config, "gpu_vram_gb", lambda: 4.0)

    assert config.resolve_profile("auto").name == "balanced"


def test_resolve_profile_unknown():
    """A typo in the profile name fails loudly."""
    with pytest.raises(ValueError, match="Unknown profile"):
        config.resolve_profile("turbo-max")


def test_resolve_settings_falls_back_to_cpu(monkeypatch):
    """Asking for CUDA without CUDA degrades to CPU rather than exiting.

    The original version called sys.exit here, which made the tool
    unusable on any machine without an NVIDIA GPU.
    """
    monkeypatch.setattr(config, "cuda_available", lambda: False)

    settings = config.resolve_settings(profile="quality", device="cuda")

    assert settings.device == "cpu"
    assert settings.compute_type == "int8"


def test_resolve_settings_uses_gpu_quantisation(monkeypatch):
    """On CUDA the profile's GPU compute type is used."""
    monkeypatch.setattr(config, "cuda_available", lambda: True)

    settings = config.resolve_settings(profile="quality", device="cuda")

    assert (settings.device, settings.compute_type) == ("cuda", "float16")


def test_default_language_is_spanish():
    """The tool is for classes in Spain."""
    assert config.resolve_settings(device="cpu").language == "es"
    assert "clase universitaria" in config.resolve_settings(device="cpu").initial_prompt


def test_language_override():
    """The language can still be switched per run."""
    settings = config.resolve_settings(language="EN", device="cpu")

    assert settings.language == "en"
    assert "university lecture" in settings.initial_prompt


def test_cpu_threads_leave_a_core_free(monkeypatch):
    """Inference must not saturate every core of a laptop."""
    monkeypatch.setattr(config.os, "cpu_count", lambda: 8)

    assert config.default_cpu_threads() == 7


def test_cpu_threads_capped(monkeypatch):
    """Thread count is capped where CTranslate2 stops scaling."""
    monkeypatch.setattr(config.os, "cpu_count", lambda: 64)

    assert config.default_cpu_threads() == 8


def test_settings_describe_mentions_key_choices():
    """The one-line summary is what gets logged at the start of a run."""
    described = config.resolve_settings(profile="low", device="cpu").describe()

    assert "profile=low" in described
    assert "device=cpu" in described


def test_cuda_available_without_torch(monkeypatch):
    """A machine with no torch installed answers False instead of raising."""
    monkeypatch.setitem(sys.modules, "torch", None)

    assert config.cuda_available() is False
    assert config.gpu_vram_gb() == 0.0


def test_cuda_available_with_broken_driver(monkeypatch):
    """A torch that raises on the CUDA query is treated as no CUDA."""
    torch = MagicMock()
    torch.cuda.is_available.side_effect = RuntimeError("driver mismatch")
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert config.cuda_available() is False


def test_gpu_vram_reported_in_gb(monkeypatch):
    """VRAM is reported in GB, from torch's byte count."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.get_device_properties.return_value.total_memory = 6 * 1024 ** 3
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert config.gpu_vram_gb() == pytest.approx(6.0)


def test_gpu_vram_zero_without_cuda(monkeypatch):
    """No CUDA device means no VRAM to report."""
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert config.gpu_vram_gb() == 0.0


def test_resolve_settings_overrides_models(monkeypatch):
    """Explicit model choices beat the profile's defaults."""
    monkeypatch.setattr(config, "cuda_available", lambda: False)

    settings = config.resolve_settings(
        profile="low", whisper_model="medium", ollama_model="llama3.2:3b"
    )

    assert settings.whisper_model == "medium"
    assert settings.ollama_model == "llama3.2:3b"


def test_resolve_settings_denoise_override():
    """Denoising can be forced off through the settings."""
    assert config.resolve_settings(device="cpu", denoise=False).denoise is False


def test_resolve_settings_unknown_device_passes_through(monkeypatch):
    """An exotic device name is respected but uses CPU quantisation."""
    monkeypatch.setattr(config, "cuda_available", lambda: False)

    settings = config.resolve_settings(profile="low", device="mps")

    assert settings.device == "mps"
    assert settings.compute_type == "int8"


def test_initial_prompt_is_none_for_unknown_language():
    """An unsupported language simply gets no biasing prompt."""
    assert config.resolve_settings(language="de", device="cpu").initial_prompt is None


def test_hallucination_patterns_are_lowercase():
    """The matcher lowercases its input, so the patterns must be lowercase."""
    assert all(pattern == pattern.lower() for pattern in config.HALLUCINATION_PATTERNS)


def test_every_profile_is_self_consistent():
    """Each profile names a model, a quantisation and an LLM."""
    for name, profile in config.PROFILES.items():
        assert profile.name == name
        assert profile.whisper_model
        assert profile.cpu_compute_type
        assert profile.gpu_compute_type
        assert profile.ollama_model
        assert profile.beam_size >= 1
