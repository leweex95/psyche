from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from .events import AudioFrame


class AudioDeviceError(RuntimeError):
    """Safe-to-display audio-device error."""


class SoundDeviceMicrophone:
    """Capture mono PCM16 frames; sounddevice is an optional voice extra."""

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

        def offer(frame: AudioFrame | BaseException) -> None:
            if queue.full():
                # Audio overflow is explicit rather than silently blocking PortAudio.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(frame)

        def callback(indata, _frames, _time_info, status):
            if status:
                loop.call_soon_threadsafe(offer, AudioDeviceError("Microphone input overflowed."))
                return
            raw = memoryview(indata).cast("B").tobytes()
            loop.call_soon_threadsafe(
                offer,
                AudioFrame(data=raw, sample_rate=self.sample_rate, captured_at=time.monotonic()),
            )

        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=samples,
                device=self.device,
                channels=1,
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
