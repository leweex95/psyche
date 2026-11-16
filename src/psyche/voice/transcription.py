"""On-device speech recognition using faster-whisper."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading

from .events import AudioFrame


class TranscriptionError(RuntimeError):
    """Safe-to-display local transcription failure."""


class LocalWhisperTranscriber:
    """Transcribe complete utterances locally; audio is never sent to a service."""

    def __init__(self, model: str = "base", model_dir: Path | None = None,
                 device: str = "cpu", compute_type: str = "int8", model_factory=None):
        self.model_name = model
        self.model_dir = model_dir or Path.home() / ".psyche" / "models" / "faster-whisper"
        self.device, self.compute_type = device, compute_type
        self._model_factory = model_factory
        self._model = None
        self._model_lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                factory = self._model_factory
                if factory is None:
                    try:
                        from faster_whisper import WhisperModel
                    except ImportError as exc:
                        raise TranscriptionError(
                            "Install local voice support with `pip install -e '.[voice]'`."
                        ) from exc
                    factory = WhisperModel
                try:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    self._model = factory(
                        self.model_name, device=self.device, compute_type=self.compute_type,
                        download_root=str(self.model_dir),
                    )
                except Exception as exc:
                    raise TranscriptionError(
                        "Could not load the local speech model. Check model download, disk space, and CPU support."
                    ) from exc
        return self._model

    def prepare(self) -> None:
        """Load the model, downloading model weights if this is the first run."""
        self._load_model()

    @staticmethod
    def _pcm_to_audio(frames: list[AudioFrame]):
        try:
            import numpy as np
        except ImportError as exc:
            raise TranscriptionError("Install local voice support with `pip install -e '.[voice]'`.") from exc
        if not frames:
            return np.empty(0, dtype=np.float32), 16000
        sample_rate = frames[0].sample_rate
        if sample_rate != 16000:
            raise TranscriptionError("Local transcription expects 16 kHz microphone audio.")
        if any(frame.sample_rate != sample_rate or frame.channels != 1 or len(frame.data) % 2 for frame in frames):
            raise TranscriptionError("Local transcription requires consistent mono PCM16 audio frames.")
        pcm = b"".join(frame.data for frame in frames)
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return audio, sample_rate

    async def transcribe(self, frames: list[AudioFrame]) -> str:
        audio, _sample_rate = self._pcm_to_audio(frames)
        if not len(audio):
            return ""
        try:
            return await asyncio.to_thread(self._transcribe_array, audio)
        except asyncio.CancelledError:
            raise
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError("Local speech recognition failed; the session remains saved.") from exc

    async def transcribe_file(self, audio_path: Path) -> str:
        """Transcribe a supported audio file locally without copying or uploading it."""
        if not audio_path.is_file():
            raise TranscriptionError("Audio file not found.")
        try:
            return await asyncio.to_thread(self._transcribe_path, audio_path)
        except asyncio.CancelledError:
            raise
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError("Local audio-file transcription failed; no transcript was stored.") from exc

    def _transcribe_array(self, audio) -> str:
        model = self._load_model()
        segments, _info = model.transcribe(
            audio, beam_size=5, vad_filter=False, condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()

    def _transcribe_path(self, audio_path: Path) -> str:
        model = self._load_model()
        segments, _info = model.transcribe(
            str(audio_path), beam_size=5, vad_filter=True, condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
