from pathlib import Path

import pytest

from psyche.config import load_config
from psyche.safety import safety_response


def test_defaults_and_environment_override(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("llm:\n  model: test-model\n", encoding="utf-8")
    monkeypatch.setenv("PSYCHE_DATA_DIR", str(tmp_path / "data"))
    config = load_config(config_file)
    assert config.llm.model == "test-model"
    assert config.data_dir == tmp_path / "data"
    assert config.conversation.target_response_words == 35
    assert config.retrieval.weights.relevance == 0.40
    assert config.context.profile_tokens == 1500
    assert config.retrieval.max_memories == 12


def test_default_voice_is_local_and_default_text_provider_is_deepseek():
    config = load_config(Path(__file__).parents[2] / "config" / "default.yaml")
    assert config.llm.provider == "deepseek"
    assert config.llm.model == "deepseek-chat"
    assert config.voice.stt_model == "base"
    assert config.voice.tts_voice == "en_US-lessac-medium"


def test_invalid_configuration_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("llm:\n  temperature: 9\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_safety_baseline():
    assert safety_response("Can you diagnose what disorder I have?")
    assert "emergency" in safety_response("I am going to kill myself tonight").lower()
    assert safety_response("I had a bad day") is None
