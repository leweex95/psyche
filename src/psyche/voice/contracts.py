from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from .events import AudioFrame


class Microphone(Protocol):
    def frames(self) -> AsyncIterator[AudioFrame]: ...


class VoiceActivityDetector(Protocol):
    async def utterances(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[list[AudioFrame]]: ...


class StreamingTranscriber(Protocol):
    async def transcribe(self, frames: list[AudioFrame]) -> str: ...


class StreamingSynthesizer(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...


class AudioPlayer(Protocol):
    async def play(self, chunks: AsyncIterator[bytes], interrupted: "asyncio.Event") -> None: ...
