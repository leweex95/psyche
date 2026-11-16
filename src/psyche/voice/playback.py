from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

try:
    import audioop
except ImportError:  # Python 3.13+ supplies this via the voice extra.
    audioop = None


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
        output_rate = self.sample_rate
        if factory is None:
            try:
                import sounddevice as sd
            except ImportError as exc:
                raise PlaybackError("Install voice support with `pip install -e '.[voice]'`.") from exc
            factory = sd.RawOutputStream
            if self.device is not None:
                try:
                    sd.check_output_settings(
                        device=self.device, channels=1, dtype="int16", samplerate=output_rate,
                    )
                except Exception:
                    try:
                        device_info = sd.query_devices(self.device, "output")
                        native_rate = int(device_info["default_samplerate"])
                        sd.check_output_settings(
                            device=self.device, channels=1, dtype="int16", samplerate=native_rate,
                        )
                        output_rate = native_rate
                    except Exception as exc:
                        raise PlaybackError(
                            "Selected output device supports neither the speech rate nor its reported default rate."
                        ) from exc
                if output_rate != self.sample_rate and audioop is None:
                    raise PlaybackError("Install voice support to resample speech for the selected output device.")
        try:
            stream = factory(samplerate=output_rate, device=self.device,
                             channels=1, dtype="int16", blocksize=0)
            resample_state = None
            with stream:
                async for chunk in chunks:
                    if interrupted.is_set():
                        break
                    if len(chunk) % 2:
                        raise PlaybackError("Speech provider returned an incomplete PCM sample.")
                    if output_rate != self.sample_rate:
                        chunk, resample_state = audioop.ratecv(
                            chunk, 2, 1, self.sample_rate, output_rate, resample_state,
                        )
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
