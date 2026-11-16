from __future__ import annotations

import asyncio
import struct
import threading
import time
from collections.abc import AsyncIterator

try:
    import audioop
except ImportError:  # Python 3.13+ supplies this via the voice extra.
    audioop = None

from .events import AudioFrame


class AudioDeviceError(RuntimeError):
    """Safe-to-display audio-device error."""


def negotiate_input_format(sd, device: int | str | None, sample_rate: int):
    """Find a supported PCM16 input format, preferring the VAD rate and full array."""
    try:
        device_info = sd.query_devices(device, "input")
        max_channels = int(device_info["max_input_channels"])
        native_rate = int(device_info["default_samplerate"])
        if max_channels < 1:
            raise AudioDeviceError("Selected input device has no microphone channels.")

        channel_options = list(dict.fromkeys((max_channels, 1)))
        rate_options = list(dict.fromkeys((sample_rate, native_rate)))
        last_format_error = None
        for rate in rate_options:
            for channels in channel_options:
                try:
                    sd.check_input_settings(
                        device=device, samplerate=rate, channels=channels, dtype="int16",
                    )
                    return device_info, rate, channels
                except Exception as exc:
                    last_format_error = exc
        raise AudioDeviceError(
            "Microphone does not support the requested or its reported default audio format."
        ) from last_format_error
    except AudioDeviceError:
        raise
    except Exception as exc:
        raise AudioDeviceError("Could not inspect the selected microphone input format.") from exc


class SoundDeviceMicrophone:
    """Capture mono PCM16 frames, adapting device-native channel/rate formats."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30,
                 device: int | str | None = None, queue_frames: int = 100):
        if frame_ms not in {10, 20, 30}:
            raise ValueError("frame_ms must be 10, 20, or 30 for WebRTC VAD")
        if sample_rate not in {8000, 16000, 32000, 48000}:
            raise ValueError("sample_rate must be supported by WebRTC VAD")
        self.sample_rate, self.frame_ms, self.device = sample_rate, frame_ms, device
        self.queue_frames = queue_frames

    async def frames(self) -> AsyncIterator[AudioFrame]:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceError("Install voice support with `pip install -e '.[voice]'`.") from exc

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[AudioFrame | BaseException] = asyncio.Queue(self.queue_frames)
        samples = self.sample_rate * self.frame_ms // 1000
        # Microphone arrays may expose their working signal only through
        # multiple channels, and WASAPI often requires the device's native
        # rate/channel count instead of the VAD's 16 kHz mono format.
        _, input_rate, input_channels = negotiate_input_format(sd, self.device, self.sample_rate)
        if input_rate != self.sample_rate and audioop is None:
            raise AudioDeviceError("Install voice support to resample microphone audio.")

        output_frame_bytes = samples * 2
        pending_pcm = bytearray()
        resample_state = None
        overflowed = threading.Event()

        def offer(frame: AudioFrame | BaseException) -> None:
            if queue.full():
                # Audio overflow is explicit rather than silently blocking PortAudio.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(frame)

        def callback(indata, _frames, _time_info, status):
            nonlocal resample_state
            if overflowed.is_set():
                return
            input_overflow = getattr(status, "input_overflow", None)
            if input_overflow is None:
                input_overflow = bool(status)
            if input_overflow:
                overflowed.set()
                loop.call_soon_threadsafe(offer, AudioDeviceError("Microphone input overflowed."))
                return
            raw = memoryview(indata).cast("B").tobytes()
            if len(raw) % (2 * input_channels):
                loop.call_soon_threadsafe(offer, AudioDeviceError("Microphone returned an incomplete PCM frame."))
                return
            if input_channels > 1:
                mono = bytearray()
                channel_format = "<" + "h" * input_channels
                for channel_samples in struct.iter_unpack(channel_format, raw):
                    sample = max(-32768, min(32767, sum(channel_samples) // input_channels))
                    mono.extend(struct.pack("<h", sample))
                raw = bytes(mono)
            if input_rate != self.sample_rate:
                raw, resample_state = audioop.ratecv(
                    raw, 2, 1, input_rate, self.sample_rate, resample_state,
                )
            pending_pcm.extend(raw)
            captured_at = time.monotonic()
            while len(pending_pcm) >= output_frame_bytes:
                frame_data = bytes(pending_pcm[:output_frame_bytes])
                del pending_pcm[:output_frame_bytes]
                loop.call_soon_threadsafe(
                    offer,
                    AudioFrame(data=frame_data, sample_rate=self.sample_rate, captured_at=captured_at),
                )

        try:
            with sd.RawInputStream(
                samplerate=input_rate,
                # Let PortAudio choose host-compatible callback sizes. PCM is
                # buffered below into fixed-size VAD frames, so callback sizes
                # need not match the VAD frame duration.
                blocksize=0,
                device=self.device,
                channels=input_channels,
                dtype="int16",
                callback=callback,
            ):
                while True:
                    item = await queue.get()
                    if isinstance(item, BaseException):
                        raise item
                    yield item
        except AudioDeviceError:
            raise
        except Exception as exc:
            raise AudioDeviceError("Could not capture microphone audio; check the selected input device.") from exc
