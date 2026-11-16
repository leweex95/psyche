from types import SimpleNamespace
import sys

import pytest

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
