from pathlib import Path
import os

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.5, ge=0, le=2)
    max_output_tokens: int = Field(default=300, ge=1, le=8192)
    request_timeout_seconds: float = Field(default=30, gt=0, le=300)


class ConversationConfig(BaseModel):
    default_methodology: str = "auto"
    target_response_words: int = Field(default=35, ge=15, le=200)


class RetrievalWeights(BaseModel):
    relevance: float = Field(default=0.40, ge=0, le=1)
    importance: float = Field(default=0.25, ge=0, le=1)
    evidence: float = Field(default=0.15, ge=0, le=1)
    recency: float = Field(default=0.10, ge=0, le=1)
    context: float = Field(default=0.10, ge=0, le=1)


class RetrievalConfig(BaseModel):
    weights: RetrievalWeights = RetrievalWeights()
    max_memories: int = Field(default=12, ge=1, le=100)


class ContextConfig(BaseModel):
    safety_instructions_tokens: int = Field(default=1000, ge=100, le=4000)
    methodology_tokens: int = Field(default=700, ge=100, le=3000)
    profile_tokens: int = Field(default=1500, ge=0, le=6000)
    retrieved_memories_tokens: int = Field(default=1500, ge=0, le=6000)
    recent_conversation_tokens: int = Field(default=3000, ge=100, le=12000)
    max_recent_messages: int = Field(default=20, ge=1, le=100)


class AppConfig(BaseModel):
    name: str = "psyche"
    log_level: str = "WARNING"


class PrivacyConfig(BaseModel):
    cloud_processing_allowed: bool = True


class SafetyConfig(BaseModel):
    enabled: bool = True


class VoiceVADConfig(BaseModel):
    enabled: bool = True
    aggressiveness: int = Field(default=2, ge=0, le=3)
    end_silence_ms: int = Field(default=550, ge=100, le=3000)
    pre_roll_ms: int = Field(default=240, ge=0, le=1000)


class VoiceConfig(BaseModel):
    input_device: str | int | None = None
    output_device: str | int | None = None
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    frame_ms: int = Field(default=30, ge=10, le=30)
    stt_model: str = "gpt-live-transcribe"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    vad: VoiceVADConfig = VoiceVADConfig()


class Config(BaseModel):
    app: AppConfig = AppConfig()
    llm: LLMConfig = LLMConfig()
    conversation: ConversationConfig = ConversationConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    context: ContextConfig = ContextConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    safety: SafetyConfig = SafetyConfig()
    voice: VoiceConfig = VoiceConfig()
    data_dir: Path = Path.home() / ".psyche"


def load_config(config_path: Path | None = None) -> Config:
    """Load YAML defaults, then apply supported environment overrides."""
    path = config_path or Path("config/default.yaml")
    values = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    values = values or {}
    if os.getenv("PSYCHE_LLM_MODEL"):
        values.setdefault("llm", {})["model"] = os.environ["PSYCHE_LLM_MODEL"]
    if os.getenv("PSYCHE_DATA_DIR"):
        values["data_dir"] = os.environ["PSYCHE_DATA_DIR"]
    result = Config.model_validate(values)
    result.data_dir = result.data_dir.expanduser()
    return result
