from typing import AsyncIterator
import os
import asyncio
from pathlib import Path

from .base import LLMMessage


class ProviderError(RuntimeError):
    """Safe-to-display provider failure; credentials and response bodies are omitted."""


class OpenAIProvider:
    def __init__(self, model: str, timeout: float = 30):
        key = os.getenv("PSYCHE_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ProviderError("No API key configured. Set PSYCHE_LLM_API_KEY in .env.")
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=key, timeout=timeout, max_retries=2)
        except Exception as exc:
            raise ProviderError("Could not initialize the LLM provider.") from exc
        self.model = model

    async def generate(self, messages: list[LLMMessage], temperature: float, max_output_tokens: int) -> str:
        try:
            result = await self.client.chat.completions.create(
                model=self.model, messages=[m.model_dump() for m in messages],
                temperature=temperature, max_tokens=max_output_tokens,
            )
            return result.choices[0].message.content or ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError("The LLM request failed or timed out. Your session was preserved; try again.") from exc

    async def stream(self, messages: list[LLMMessage], temperature: float, max_output_tokens: int) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.model, messages=[m.model_dump() for m in messages],
                temperature=temperature, max_tokens=max_output_tokens, stream=True,
            )
            async for event in stream:
                delta = event.choices[0].delta.content
                if delta:
                    yield delta
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError("The LLM stream failed. Your session was preserved; try again.") from exc

    async def generate_structured(self, messages: list[LLMMessage], response_model):
        try:
            result = await self.client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=0,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError("The structured LLM request failed or timed out.") from exc
        try:
            text = result.choices[0].message.content or "{}"
            return response_model.model_validate_json(text)
        except Exception as exc:
            raise ProviderError("The provider returned an invalid structured response.") from exc

    async def transcribe_audio(self, audio_path: Path, model: str = "whisper-1") -> str:
        try:
            with audio_path.open("rb") as audio_file:
                result = await self.client.audio.transcriptions.create(
                    model=model, file=audio_file, response_format="text",
                )
            if isinstance(result, str):
                return result.strip()
            return (getattr(result, "text", "") or "").strip()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError("Audio transcription failed or timed out; the audio file was not stored locally.") from exc
