from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
from collections.abc import AsyncIterator, Callable

from .events import AudioFrame

try:
    import audioop
except ImportError:  # Python 3.13+ supplies this via the voice extra.
    audioop = None

_REALTIME_SAMPLE_RATE = 24000


class TranscriptionError(RuntimeError):
    """Safe-to-display live transcription failure."""


class OpenAIRealtimeTranscriber:
    """Incrementally stream active VAD frames to Realtime transcription."""

    def __init__(self, model: str = "gpt-live-transcribe", timeout: float = 30,
                 on_partial: Callable[[str], None] | None = None, connector=None):
        self.model, self.timeout = model, timeout
        self.on_partial = on_partial
        self._connector = connector

    async def transcribe(self, frames: list[AudioFrame]) -> str:
        if not frames:
            return ""
        async def frame_stream():
            for frame in frames:
                yield frame
        return await self.transcribe_stream(frame_stream())

    async def transcribe_stream(
        self, frames: AsyncIterator[AudioFrame], on_partial: Callable[[str], object] | None = None,
    ) -> str:
        key = os.getenv("PSYCHE_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise TranscriptionError("No API key configured for speech transcription.")
        frame_iterator = frames.__aiter__()
        try:
            first_frame = await anext(frame_iterator)
        except StopAsyncIteration:
            return ""

        source_rate = first_frame.sample_rate
        if source_rate not in {16000, _REALTIME_SAMPLE_RATE}:
            raise TranscriptionError("Realtime transcription input must be 16 or 24 kHz PCM16")
        if first_frame.channels != 1 or len(first_frame.data) % 2:
            raise TranscriptionError("Realtime transcription requires mono PCM16 frames")
        if source_rate != _REALTIME_SAMPLE_RATE and audioop is None:
            raise TranscriptionError("Install voice support to resample microphone audio for realtime transcription.")

        async def all_frames():
            yield first_frame
            async for frame in frame_iterator:
                yield frame

        connector = self._connector
        if connector is None:
            try:
                from websockets.asyncio.client import connect
            except ImportError as exc:
                raise TranscriptionError("Install voice support with `pip install -e '.[voice]'`.") from exc
            connector = connect

        url = "wss://api.openai.com/v1/realtime?intent=transcription"
        headers = {"Authorization": f"Bearer {key}"}
        try:
            async with asyncio.timeout(self.timeout):
                async with connector(url, additional_headers=headers, open_timeout=self.timeout) as ws:
                    await ws.send(json.dumps({
                        "type": "session.update",
                        "session": {
                            "type": "transcription",
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": _REALTIME_SAMPLE_RATE},
                                    "transcription": {"model": self.model, "delay": "low"},
                                    "turn_detection": None,
                                }
                            },
                        },
                    }))
                    await self._wait_for(ws, {"session.updated"})

                    async def send_audio() -> None:
                        rate_state = None
                        async for frame in all_frames():
                            if (frame.sample_rate != source_rate or frame.channels != 1
                                    or len(frame.data) % 2):
                                raise ValueError("Realtime transcription requires consistent mono PCM16 frames")
                            pcm = frame.data
                            if source_rate != _REALTIME_SAMPLE_RATE:
                                pcm, rate_state = audioop.ratecv(
                                    pcm, 2, 1, source_rate, _REALTIME_SAMPLE_RATE, rate_state,
                                )
                            await ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(pcm).decode("ascii"),
                            }))
                        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                    sender = asyncio.create_task(send_audio())

                    async def receive_event() -> dict:
                        receiving = asyncio.create_task(ws.recv())
                        if not sender.done():
                            done, _ = await asyncio.wait(
                                {sender, receiving}, return_when=asyncio.FIRST_COMPLETED
                            )
                            if sender in done and sender.exception() is not None:
                                receiving.cancel()
                                await asyncio.gather(receiving, return_exceptions=True)
                                await sender
                        raw = await receiving
                        return json.loads(raw)

                    try:
                        while True:
                            event = await receive_event()
                            event_type = event.get("type")
                            if event_type == "conversation.item.input_audio_transcription.delta":
                                callback = on_partial or self.on_partial
                                if callback:
                                    result = callback(event.get("delta", ""))
                                    if inspect.isawaitable(result):
                                        await result
                            elif event_type == "conversation.item.input_audio_transcription.completed":
                                await sender
                                return event.get("transcript", "")
                            elif event_type in {
                                "conversation.item.input_audio_transcription.failed", "error"
                            }:
                                self._raise_provider_error(event, initializing=False)
                    finally:
                        if not sender.done():
                            sender.cancel()
                        await asyncio.gather(sender, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except TranscriptionError:
            raise
        except TimeoutError as exc:
            raise TranscriptionError("Speech transcription timed out; your session is preserved.") from exc
        except Exception as exc:
            raise TranscriptionError("Could not transcribe speech; check network/provider availability.") from exc

    @staticmethod
    async def _wait_for(ws, expected: set[str]) -> dict:
        while True:
            event = json.loads(await ws.recv())
            event_type = event.get("type")
            if event_type in expected:
                return event
            if event_type == "error":
                OpenAIRealtimeTranscriber._raise_provider_error(event, initializing=True)

    @staticmethod
    def _raise_provider_error(event: dict, *, initializing: bool) -> None:
        details = event.get("error")
        if not isinstance(details, dict):
            details = {}
        code = details.get("code")
        if code in {"insufficient_quota", "credit_balance_exhausted"}:
            raise TranscriptionError("Speech transcription is unavailable: the provider account has no credits.")
        if code == "invalid_api_key":
            raise TranscriptionError("Speech transcription is unavailable: the provider rejected the configured credentials.")
        message = (
            "Speech transcription session could not be initialized."
            if initializing else "Speech transcription failed; retry the turn."
        )
        raise TranscriptionError(message)
