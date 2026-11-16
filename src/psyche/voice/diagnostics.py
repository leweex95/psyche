from __future__ import annotations

import math
import struct
import threading

from .audio import AudioDeviceError, negotiate_input_format


def _level_summary(count: int, total_squares: int, peak: int, nonzero: int, clipped: int) -> dict:
    rms = math.sqrt(total_squares / count) if count else 0.0
    dbfs = 20 * math.log10(rms / 32768) if rms else None
    return {
        "samples": count,
        "peak": peak,
        "rms": round(rms, 1),
        "dbfs": round(dbfs, 1) if dbfs is not None else None,
        "nonzero": nonzero,
        "clipped": clipped,
    }


def capture_input_levels(sd, device: int | str | None, target_rate: int, seconds: float) -> dict:
    """Measure microphone levels in memory; never return or retain PCM audio."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    device_info, input_rate, channels = negotiate_input_format(sd, device, target_rate)
    target_frames = max(1, round(seconds * input_rate))
    blocksize = max(1, input_rate // 10)
    channel_squares = [0] * channels
    channel_peaks = [0] * channels
    channel_nonzero = [0] * channels
    channel_clipped = [0] * channels
    mixed_squares = mixed_peak = mixed_nonzero = mixed_clipped = 0
    captured_frames = 0
    statuses: list[str] = []
    callback_error: list[str] = []
    finished = threading.Event()

    def callback(indata, frames, _time_info, status):
        nonlocal captured_frames, mixed_squares, mixed_peak, mixed_nonzero, mixed_clipped
        if status:
            statuses.append(str(status))
        raw = memoryview(indata).cast("B")
        expected_bytes = frames * channels * 2
        if len(raw) != expected_bytes:
            callback_error.append("Microphone returned an incomplete PCM frame.")
            finished.set()
            return

        sample_format = "<" + "h" * channels
        for row in struct.iter_unpack(sample_format, raw):
            mixed = max(-32768, min(32767, sum(row) // channels))
            mixed_squares += mixed * mixed
            mixed_peak = max(mixed_peak, abs(mixed))
            mixed_nonzero += mixed != 0
            mixed_clipped += abs(mixed) >= 32760
            for index, value in enumerate(row):
                channel_squares[index] += value * value
                channel_peaks[index] = max(channel_peaks[index], abs(value))
                channel_nonzero[index] += value != 0
                channel_clipped[index] += abs(value) >= 32760
        captured_frames += frames
        if captured_frames >= target_frames:
            finished.set()

    try:
        with sd.RawInputStream(
            samplerate=input_rate,
            blocksize=blocksize,
            device=device,
            channels=channels,
            dtype="int16",
            callback=callback,
        ):
            if not finished.wait(seconds + 5):
                raise AudioDeviceError("Timed out waiting for microphone samples.")
    except AudioDeviceError:
        raise
    except Exception as exc:
        raise AudioDeviceError("Could not capture microphone levels; check the selected input device.") from exc

    if callback_error:
        raise AudioDeviceError(callback_error[0])
    return {
        "device": device_info,
        "sample_rate": input_rate,
        "channels": channels,
        "frames": captured_frames,
        "seconds": captured_frames / input_rate,
        "statuses": statuses,
        "channel_levels": [
            _level_summary(captured_frames, channel_squares[index], channel_peaks[index],
                           channel_nonzero[index], channel_clipped[index])
            for index in range(channels)
        ],
        "mixed_level": _level_summary(
            captured_frames, mixed_squares, mixed_peak, mixed_nonzero, mixed_clipped,
        ),
    }
