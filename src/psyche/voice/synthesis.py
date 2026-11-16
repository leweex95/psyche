"""On-device speech synthesis using Piper voices."""

from __future__ import annotations

import asyncio
from pathlib import Path

try:
    import audioop
except ImportError:  # Python 3.13+ supplies this through the voice extra.
    audioop = None


class SynthesisError(RuntimeError):
    """Safe-to-display local speech synthesis failure."""


class LocalPiperSynthesizer:
    """Synthesize reply audio locally and yield mono PCM16 at the playback rate."""

    def __init__(self, voice: str = "en_US-lessac-medium", model_dir: Path | None = None,
                 output_sample_rate: int = 24000, voice_loader=None, downloader=None):
        self.voice_name = voice
        self.model_dir = model_dir or Path.home() / ".psyche" / "models" / "piper"
        self.output_sample_rate = output_sample_rate
        self._voice_loader, self._downloader = voice_loader, downloader
        self._voice = None

    def _load_voice(self):
        if self._voice is not None:
            return self._voice
        model_path = self.model_dir / f"{self.voice_name}.onnx"
        config_path = self.model_dir / f"{self.voice_name}.onnx.json"
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            if not model_path.is_file() or not config_path.is_file():
                downloader = self._downloader
                if downloader is None:
                    from piper.download_voices import download_voice
                    downloader = download_voice
                downloader(self.voice_name, self.model_dir)
            loader = self._voice_loader
            if loader is None:
                from piper import PiperVoice
                loader = PiperVoice.load
            self._voice = loader(model_path, config_path=config_path, use_cuda=False)
            return self._voice
        except ImportError as exc:
            raise SynthesisError("Install local voice support with `pip install -e '.[voice]'`.") from exc
        except Exception as exc:
            raise SynthesisError(
                "Could not load the local speech voice. Check voice name, download, and disk space."
            ) from exc

    def prepare(self) -> None:
        """Download voice weights if needed and initialize the local synthesizer."""
        self._load_voice()

    def _synthesize_bytes(self, text: str) -> bytes:
        voice = self._load_voice()
        chunks = []
        source_rate = None
        rate_state = None
        for chunk in voice.synthesize(text):
            if chunk.sample_width != 2 or chunk.sample_channels != 1:
                raise SynthesisError("The selected local Piper voice did not produce mono PCM16 audio.")
            if source_rate is None:
                source_rate = chunk.sample_rate
            elif source_rate != chunk.sample_rate:
                raise SynthesisError("The local Piper voice returned inconsistent sample rates.")
            data = chunk.audio_int16_bytes
            if chunk.sample_rate != self.output_sample_rate:
                if audioop is None:
                    raise SynthesisError("Install the voice extra to resample local speech for playback.")
                data, rate_state = audioop.ratecv(
                    data, 2, 1, chunk.sample_rate, self.output_sample_rate, rate_state,
                )
            chunks.append(data)
        return b"".join(chunks)

    async def synthesize(self, text: str):
        if not text.strip():
            return
        try:
            pcm = await asyncio.to_thread(self._synthesize_bytes, text)
        except asyncio.CancelledError:
            raise
        except SynthesisError:
            raise
        except Exception as exc:
            raise SynthesisError("Local speech synthesis failed; the text response remains saved.") from exc
        # Bound queue/playback chunk size to keep barge-in responsive even though
        # Piper renders each short sentence locally before playback begins.
        for offset in range(0, len(pcm), 4096):
            chunk = pcm[offset:offset + 4096]
            if len(chunk) % 2:
                chunk = chunk[:-1]
            if chunk:
                yield chunk
