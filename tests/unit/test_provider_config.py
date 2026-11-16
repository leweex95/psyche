from types import SimpleNamespace
import sys
import traceback

import pytest
from pydantic import BaseModel

from psyche.llm.cloud import OpenAICompatibleProvider, ProviderError, provider_for


def test_deepseek_provider_uses_deepseek_endpoint_and_api_key(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "local-test-secret")
    config = SimpleNamespace(provider="deepseek", model="deepseek-chat", request_timeout_seconds=17)

    provider = provider_for(config)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "deepseek-chat"
    assert captured["api_key"] == "local-test-secret"
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["timeout"] == 17


def test_deepseek_provider_requires_its_configured_credential(monkeypatch):
    monkeypatch.delenv("PSYCHE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = SimpleNamespace(provider="deepseek", model="deepseek-chat", request_timeout_seconds=10)

    with pytest.raises(ProviderError, match="DEEPSEEK_API_KEY"):
        provider_for(config)


def test_provider_initialization_does_not_chain_sdk_error(monkeypatch):
    secret = "credential-visible-only-to-fake-sdk"

    class FailingAsyncOpenAI:
        def __init__(self, **_kwargs):
            raise RuntimeError(f"failed to initialize using {secret}")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FailingAsyncOpenAI))
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    config = SimpleNamespace(provider="deepseek", model="deepseek-chat", request_timeout_seconds=10)

    with pytest.raises(ProviderError) as caught:
        provider_for(config)

    rendered = "".join(traceback.format_exception(caught.value))
    assert secret not in str(caught.value)
    assert secret not in rendered
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["generate", "stream", "structured"])
async def test_provider_failures_do_not_expose_sdk_exception_text_or_chain(operation):
    secret = "Bearer-SUPER-SECRET-credential"

    async def fail_request(**_kwargs):
        raise RuntimeError(f"request failed with Authorization: {secret}")

    provider = object.__new__(OpenAICompatibleProvider)
    provider.model = "deepseek-chat"
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail_request))
    )
    messages = []

    with pytest.raises(ProviderError) as caught:
        if operation == "generate":
            await provider.generate(messages, 0.5, 100)
        elif operation == "stream":
            _ = [delta async for delta in provider.stream(messages, 0.5, 100)]
        else:
            class Result(BaseModel):
                value: str

            await provider.generate_structured(messages, Result)

    rendered = "".join(traceback.format_exception(caught.value))
    assert secret not in str(caught.value)
    assert secret not in rendered
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_invalid_structured_response_does_not_chain_provider_content():
    secret = "PRIVATE-RESPONSE-CONTENT"

    class Result(BaseModel):
        value: int

    async def return_invalid(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f'{{"value":"{secret}"}}'))])

    provider = object.__new__(OpenAICompatibleProvider)
    provider.model = "deepseek-chat"
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=return_invalid))
    )

    with pytest.raises(ProviderError) as caught:
        await provider.generate_structured([], Result)

    rendered = "".join(traceback.format_exception(caught.value))
    assert secret not in str(caught.value)
    assert secret not in rendered
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
