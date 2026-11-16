from typing import Protocol, AsyncIterator
from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMProvider(Protocol):
    async def generate(self, messages: list[LLMMessage], temperature: float, max_output_tokens: int) -> str: ...
    def stream(self, messages: list[LLMMessage], temperature: float, max_output_tokens: int) -> AsyncIterator[str]: ...
    async def generate_structured(self, messages: list[LLMMessage], response_model: type[BaseModel]) -> BaseModel: ...
