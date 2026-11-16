from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


class PlaybackError(RuntimeError):
    """Safe-to-display audio playback failure."""


class SoundDeviceAudioPlayer:
    """Play mono PCM16 chunks and stop between bounded chunks on interruption."""

    def __init__(self, sample_rate: int = 24000, device: int | str | None = None,
                 stream_factory=None):
        self.sample_rate, self.device = sample_rate, device
        self._stream_factory = stream_factory

    async def play(self, chunks: AsyncIterator[bytes], interrupted: asyncio.Event) -> None:
        factory = self._stream_factory
        if factory is None:
            try:
                import sounddevice as sd
            except ImportError as exc:
                raise PlaybackError("Install voice support with `pip install -e '.[voice]'`.") from exc
            factory = sd.RawOutputStream
        try:
            stream = factory(samplerate=self.sample_rate, device=self.device,
                             channels=1, dtype="int16", blocksize=0)
            with stream:
                async for chunk in chunks:
                    if interrupted.is_set():
                        break
                    if len(chunk) % 2:
                        raise PlaybackError("Speech provider returned an incomplete PCM sample.")
                    # A bounded chunk (normally 4096 bytes, ~85 ms at 24 kHz)
                    # limits interruption latency without blocking the event loop.
                    await asyncio.to_thread(stream.write, chunk)
                    if interrupted.is_set():
                        break
        except asyncio.CancelledError:
            raise
        except PlaybackError:
            raise
        except Exception as exc:
            raise PlaybackError("Could not play audio; check the selected output device.") from exc
