from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class VoiceState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    CAPTURING = "capturing"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTING = "interrupting"
    ERROR = "error"


class VoiceEventType(StrEnum):
    SPEECH_STARTED = "speech_started"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    GENERATION_STARTED = "generation_started"
    RESPONSE_TEXT_DELTA = "response_text_delta"
    AUDIO_CHUNK = "audio_chunk"
    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_INTERRUPTED = "playback_interrupted"
    TURN_COMPLETED = "turn_completed"
    ERROR = "error"


class VoiceEvent(BaseModel):
    """Metadata-only event. Audio bytes travel on a separate typed channel."""

    type: VoiceEventType
    session_id: str
    turn_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AudioFrame(BaseModel):
    """One PCM frame; intentionally not serializable as a VoiceEvent payload."""

    data: bytes
    sample_rate: int = Field(gt=0)
    channels: int = Field(default=1, gt=0)
    captured_at: float


class StateTransitionError(RuntimeError):
    pass


class VoiceStateMachine:
    _allowed = {
        VoiceState.IDLE: {VoiceState.LISTENING, VoiceState.ERROR},
        VoiceState.LISTENING: {VoiceState.CAPTURING, VoiceState.ERROR, VoiceState.IDLE},
        VoiceState.CAPTURING: {VoiceState.TRANSCRIBING, VoiceState.LISTENING, VoiceState.ERROR},
        VoiceState.TRANSCRIBING: {VoiceState.THINKING, VoiceState.LISTENING, VoiceState.ERROR},
        VoiceState.THINKING: {VoiceState.SPEAKING, VoiceState.LISTENING, VoiceState.ERROR},
        VoiceState.SPEAKING: {VoiceState.INTERRUPTING, VoiceState.LISTENING, VoiceState.ERROR},
        VoiceState.INTERRUPTING: {VoiceState.LISTENING, VoiceState.ERROR},
        VoiceState.ERROR: {VoiceState.IDLE},
    }

    def __init__(self) -> None:
        self.state = VoiceState.IDLE
        self.history: list[tuple[VoiceState, VoiceState]] = []

    def transition(self, target: VoiceState) -> None:
        if target not in self._allowed[self.state]:
            raise StateTransitionError(f"Invalid voice transition: {self.state} -> {target}")
        previous = self.state
        self.state = target
        self.history.append((previous, target))
