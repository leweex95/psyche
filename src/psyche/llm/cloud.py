from typing import AsyncIterator
import os
import asyncio

from .base import LLMMessage


class ProviderError(RuntimeError):
    """Safe-to-display provider failure; credentials and response bodies are omitted."""


class OpenAICompatibleProvider:
    def __init__(self, model: str, timeout: float = 30, *, base_url: str | None = None,
                 key_env: str = "OPENAI_API_KEY", provider_name: str = "OpenAI"):
        key = os.getenv(key_env)
        if not key:
            raise ProviderError(f"No {provider_name} API key configured. Set {key_env} in .env.")
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=2)
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

def provider_for(config, timeout: float | None = None):
    """Build the configured text-only conversation/extraction provider."""
    provider = config.provider.casefold()
    request_timeout = timeout if timeout is not None else config.request_timeout_seconds
    if provider == "deepseek":
        return OpenAICompatibleProvider(
            config.model, request_timeout, base_url="https://api.deepseek.com",
            key_env="DEEPSEEK_API_KEY", provider_name="DeepSeek",
        )
    if provider == "openai":
        return OpenAICompatibleProvider(config.model, request_timeout)
    raise ProviderError(f"Unsupported text LLM provider: {config.provider!r}.")
