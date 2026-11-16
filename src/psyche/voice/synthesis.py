from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator


class SynthesisError(RuntimeError):
    """Safe-to-display speech synthesis failure."""


class OpenAITTSSynthesizer:
    """Yield OpenAI PCM speech response bytes as they arrive from the provider."""

    def __init__(self, model: str, voice: str, timeout: float = 30, client=None):
        self.model, self.voice, self.timeout = model, voice, timeout
        self._client = client

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        client = self._client
        if client is None:
            key = os.getenv("PSYCHE_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not key:
                raise SynthesisError("No API key configured for speech synthesis.")
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=key, timeout=self.timeout)
            except Exception as exc:
                raise SynthesisError("Could not initialize speech synthesis.") from exc
            self._client = client
        try:
            async with asyncio.timeout(self.timeout):
                async with client.audio.speech.with_streaming_response.create(
                    model=self.model,
                    voice=self.voice,
                    input=text,
                    response_format="pcm",
                ) as response:
                    async for chunk in response.iter_bytes(chunk_size=4096):
                        if chunk:
                            yield chunk
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise SynthesisError("Speech synthesis timed out; the text response is preserved.") from exc
        except Exception as exc:
            raise SynthesisError("Speech synthesis failed; retry the response audio.") from exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
