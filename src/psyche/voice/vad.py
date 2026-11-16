from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from .events import AudioFrame


class SpeechClassifier(Protocol):
    def is_speech(self, pcm: bytes, sample_rate: int) -> bool: ...


@dataclass(frozen=True)
class SpeechStarted:
    """VAD onset marker, emitted before the completed utterance."""
    initial_frames: list[AudioFrame]


@dataclass(frozen=True)
class SpeechFrame:
    """One frame belonging to the active VAD segment, including end silence."""
    frame: AudioFrame


@dataclass(frozen=True)
class UtteranceCompleted:
    frames: list[AudioFrame]
    reason: str = "silence"


class WebRtcSpeechClassifier:
    def __init__(self, aggressiveness: int = 2):
        if aggressiveness not in range(4):
            raise ValueError("aggressiveness must be between 0 and 3")
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError("Install voice support with `pip install -e '.[voice]'`.") from exc
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        return self._vad.is_speech(pcm, sample_rate)


class UtteranceSegmenter:
    """Frame-aligned local VAD with pre-roll and configurable end silence."""

    def __init__(self, classifier: SpeechClassifier, sample_rate: int = 16000,
                 frame_ms: int = 30, end_silence_ms: int = 550,
                 pre_roll_ms: int = 240, min_speech_ms: int = 120):
        if end_silence_ms < frame_ms or pre_roll_ms < 0 or min_speech_ms < frame_ms:
            raise ValueError("Invalid VAD timing configuration")
        self.classifier = classifier
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.end_silence_frames = max(1, end_silence_ms // frame_ms)
        self.pre_roll_frames = pre_roll_ms // frame_ms
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)

    async def events(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[SpeechStarted | UtteranceCompleted]:
        pre_roll: deque[AudioFrame] = deque(maxlen=self.pre_roll_frames)
        active: list[AudioFrame] | None = None
        silence = 0
        speech_frames = 0
        async for frame in frames:
            if frame.sample_rate != self.sample_rate or frame.channels != 1:
                raise ValueError("VAD requires mono frames at the configured sample rate")
            is_speech = self.classifier.is_speech(frame.data, frame.sample_rate)
            if active is None:
                if is_speech:
                    active = list(pre_roll)
                    active.append(frame)
                    speech_frames = 1
                    silence = 0
                    yield SpeechStarted(list(pre_roll))
                    yield SpeechFrame(frame)
                else:
                    pre_roll.append(frame)
            else:
                active.append(frame)
                yield SpeechFrame(frame)
                if is_speech:
                    speech_frames += 1
                    silence = 0
                else:
                    silence += 1
                    if silence >= self.end_silence_frames:
                        if speech_frames >= self.min_speech_frames:
                            yield UtteranceCompleted(active)
                        active, silence, speech_frames = None, 0, 0
                        pre_roll.clear()
        if active is not None and speech_frames >= self.min_speech_frames:
            yield UtteranceCompleted(active, reason="stream_end")

    async def utterances(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[list[AudioFrame]]:
        async for event in self.events(frames):
            if isinstance(event, UtteranceCompleted):
                yield event.frames
