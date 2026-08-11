"""M1 — Audio capture module.

Records from the microphone straight to disk, block by block. Nothing is
buffered in RAM beyond a one-second chunk, so a two-hour lecture costs a
few kilobytes of memory instead of half a gigabyte — and if the laptop
dies mid-class, everything recorded up to that point is already on disk
and perfectly playable.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

import config
from modules.audio_utils import dbfs, peak_dbfs, to_mono

logger = logging.getLogger(__name__)

# Populated lazily by `_get_sd()`. Importing sounddevice pulls in PortAudio,
# which is not available on headless machines (and not needed to run tests).
sd = None


def _get_sd():
    """Imports sounddevice on first use and caches it."""
    global sd
    if sd is None:
        import sounddevice as _sd

        sd = _sd
    return sd


def list_input_devices() -> list[dict]:
    """Returns the available input devices as dicts.

    Each entry has `index`, `name`, `channels` and `default` keys. Useful
    for picking an external microphone: a lapel or directional mic aimed
    at the professor beats any amount of software denoising.
    """
    audio = _get_sd()
    devices = audio.query_devices()
    try:
        default_index = audio.default.device[0]
    except Exception:       # pragma: no cover - depends on the host
        default_index = None

    result = []
    for index, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        result.append(
            {
                "index": index,
                "name": dev.get("name", "unknown"),
                "channels": dev.get("max_input_channels", 0),
                "default": index == default_index,
            }
        )
    return result


def record(
    output_path: Path,
    duration_seconds: int | None = None,
    sample_rate: int = config.SAMPLE_RATE,
    channels: int = config.CHANNELS,
    device: int | str | None = None,
    subtype: str = config.RECORD_SUBTYPE,
    stop_event: threading.Event | None = None,
    on_level: Callable[[float, float, float], None] | None = None,
) -> Path:
    """Records audio from the microphone, streaming it to a WAV file.

    Recording stops when `duration_seconds` elapses, when `stop_event` is
    set, or when the user presses Ctrl+C — every one of those is a normal
    way to end a class, so the partial recording is kept and returned
    rather than discarded.

    Args:
        output_path: Destination path for the WAV file.
        duration_seconds: Maximum length in seconds, or None to record
            until stopped.
        sample_rate: Sample rate in Hz. Default 16000.
        channels: Number of channels to capture. Default 1 (mono).
        device: Input device index or name. None uses the system default.
        subtype: WAV encoding. PCM_16 halves file size versus float32.
        stop_event: A `threading.Event` that ends the recording when set.
            This is how a Stop button works, since a GUI cannot deliver a
            KeyboardInterrupt to the worker thread.
        on_level: Called once per block with
            `(elapsed_seconds, level_dbfs, peak_dbfs)`. Used to drive a
            live level meter; exceptions raised by it never interrupt the
            recording.

    Returns:
        The path to the saved WAV file.

    Raises:
        FileNotFoundError: If the output directory does not exist.
        RuntimeError: If no input device is available.
    """
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_path.parent}"
        )

    audio = _get_sd()

    try:
        device_info = audio.query_devices(device, kind="input") if device is not None \
            else audio.query_devices(kind="input")
    except Exception as exc:
        raise RuntimeError("No input audio device found.") from exc

    name = device_info["name"] if isinstance(device_info, dict) else str(device_info)
    logger.info("Recording device: %s", name)
    logger.info(
        "Recording at %d Hz, %d channel(s) — %s. Press Ctrl+C to stop early.",
        sample_rate,
        channels,
        f"{duration_seconds} s max" if duration_seconds else "no time limit",
    )

    block_frames = max(1, int(config.RECORD_BLOCK_SECONDS * sample_rate))
    target_frames = int(duration_seconds * sample_rate) if duration_seconds else None

    frames_written = 0
    started = time.monotonic()
    next_report = config.LEVEL_LOG_INTERVAL
    window: list[float] = []
    window_peak = -np.inf

    try:
        with sf.SoundFile(
            str(output_path),
            mode="w",
            samplerate=sample_rate,
            channels=channels,
            subtype=subtype,
        ) as handle:
            with audio.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                device=device,
                blocksize=block_frames,
            ) as stream:
                while target_frames is None or frames_written < target_frames:
                    if stop_event is not None and stop_event.is_set():
                        logger.info(
                            "Recording stopped on request after %.0f s",
                            frames_written / sample_rate,
                        )
                        break

                    to_read = block_frames
                    if target_frames is not None:
                        to_read = min(block_frames, target_frames - frames_written)

                    chunk, overflowed = stream.read(to_read)
                    if overflowed:
                        logger.warning("Input overflow — some audio may be missing")

                    chunk = np.asarray(chunk, dtype=np.float32)
                    if chunk.size == 0:
                        break

                    handle.write(chunk)
                    frames_written += len(chunk)

                    mono = to_mono(chunk)
                    block_level = dbfs(mono)
                    block_peak = peak_dbfs(mono)
                    window.append(block_level)
                    window_peak = max(window_peak, block_peak)

                    elapsed = frames_written / sample_rate

                    if on_level is not None:
                        # A broken meter must never cost the user the class.
                        try:
                            on_level(elapsed, block_level, block_peak)
                        except Exception:       # noqa: BLE001
                            logger.exception("Level callback failed — continuing to record")

                    if elapsed >= next_report:
                        _report_level(elapsed, target_frames, sample_rate, window, window_peak)
                        window = []
                        window_peak = -np.inf
                        next_report += config.LEVEL_LOG_INTERVAL

    except KeyboardInterrupt:
        logger.info("Recording stopped by user after %.0f s", frames_written / sample_rate)

    wall_clock = time.monotonic() - started
    logger.info(
        "Recording saved to %s (%.1f s of audio, %.1f s wall clock)",
        output_path,
        frames_written / sample_rate,
        wall_clock,
    )

    if frames_written == 0:
        logger.warning("No audio was captured — check that the microphone is not muted")

    return output_path


def _report_level(
    elapsed: float,
    target_frames: int | None,
    sample_rate: int,
    window: list[float],
    window_peak: float,
) -> None:
    """Logs progress plus the input level, and flags obvious mic problems."""
    average = float(np.mean(window)) if window else -np.inf
    total = f" / {target_frames / sample_rate:.0f}" if target_frames else ""
    logger.info(
        "Recorded %.0f%s s — level %.1f dBFS (peak %.1f dBFS)",
        elapsed,
        total,
        average,
        window_peak,
    )

    if window_peak < -45.0:
        logger.warning(
            "Input is very quiet (peak %.1f dBFS). Move the microphone closer "
            "to the professor or raise the input gain.",
            window_peak,
        )
    elif window_peak > -1.0:
        logger.warning(
            "Input is clipping (peak %.1f dBFS). Lower the input gain.",
            window_peak,
        )
