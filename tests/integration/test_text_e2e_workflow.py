import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from psyche import cli
from psyche.memory.extractor import ExtractedMemory, ExtractionResult
from psyche.storage.database import Database


class FakeExtractionProvider:
    def __init__(self, *_args):
        pass

    async def generate_structured(self, _messages, _response_model):
        quote = "I feel grounded by taking a short walk after difficult workdays"
        return ExtractionResult(items=[ExtractedMemory(
            statement=quote + ".",
            evidence_quote=quote,
            category="coping_behavior",
            importance=4,
        )])


class FakeConversationProvider:
    calls = []

    def __init__(self, *_args):
        pass

    async def generate(self, messages, _temperature, _max_output_tokens):
        self.calls.append(messages)
        return "That sounds like a useful moment of steadiness to notice."


@pytest.mark.skipif(sys.platform != "win32", reason="Encrypted storage requires Windows DPAPI")
def test_text_e2e_import_review_correct_recall_and_delete(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    config = SimpleNamespace(
        data_dir=data_dir,
        privacy=SimpleNamespace(cloud_processing_allowed=True),
        llm=SimpleNamespace(
            model="test-model", request_timeout_seconds=2,
            temperature=0.5, max_output_tokens=100,
        ),
        safety=SimpleNamespace(enabled=False),
        retrieval=SimpleNamespace(weights=None, max_memories=12),
        context=SimpleNamespace(
            safety_instructions_tokens=1000, methodology_tokens=700,
            profile_tokens=1500, retrieved_memories_tokens=1500,
            recent_conversation_tokens=3000, max_recent_messages=20,
        ),
    )
    monkeypatch.setattr(cli, "services", lambda: (config, Database(data_dir)))
    runner = CliRunner()

    # Synthetic long-form source: exactly 3,000 whitespace-delimited words.
    target = "I feel grounded by taking a short walk after difficult workdays"
    essay = " ".join(["context"] * (3000 - len(target.split())) + target.split()) + "."
    assert len(essay.split()) == 3000
    source_file = tmp_path / "personal-essay.txt"
    source_file.write_text(essay, encoding="utf-8")

    imported = runner.invoke(cli.app, ["import", str(source_file)])
    assert imported.exit_code == 0, imported.output
    db = Database(data_dir)
    document_id = db.documents()[0][0]
    assert db.get_document(document_id)[2] == essay
    db.close()

    monkeypatch.setattr(cli, "provider_for", lambda _config: FakeExtractionProvider())
    extracted = runner.invoke(cli.app, ["memory", "extract", "--document-id", document_id])
    assert extracted.exit_code == 0, extracted.output
    assert "1 pending candidates added" in extracted.output

    # Both the conservative verbatim candidate and extracted candidate await review.
    db = Database(data_dir)
    pending = db.list_memories("pending")
    assert len(pending) == 2
    extracted_id = next(row[0] for row in pending if row[2] == target + ".")
    assert db.get_memory_sources(extracted_id)[0][2].startswith("char:")
    db.close()

    reviews = iter(["approve", "approve"])
    monkeypatch.setattr(cli.typer, "prompt", lambda *_args, **_kwargs: next(reviews))
    reviewed = runner.invoke(cli.app, ["memory", "review"])
    assert reviewed.exit_code == 0, reviewed.output

    corrected = "I feel steadier after taking a short walk following demanding workdays."
    correction = runner.invoke(cli.app, ["memory", "correct", extracted_id, corrected])
    assert correction.exit_code == 0, correction.output
    profile_result = runner.invoke(cli.app, ["memory", "profile"])
    assert profile_result.exit_code == 0, profile_result.output
    assert corrected in " ".join(profile_result.output.split())

    FakeConversationProvider.calls = []
    monkeypatch.setattr(cli, "provider_for", lambda _config: FakeConversationProvider())
    prompts = iter([
        "What helps me feel grounded after work?",
        "/method", "cbt",
        "What helps me feel grounded after work?",
        "EOF",
    ])

    def session_prompt(*_args, **_kwargs):
        value = next(prompts)
        if value == "EOF":
            raise EOFError
        return value

    monkeypatch.setattr(cli.typer, "prompt", session_prompt)
    conversation = runner.invoke(cli.app, ["session", "start", "--method", "auto"])
    assert conversation.exit_code == 0, conversation.output
    assert "Session left open and saved" in conversation.output
    assert len(FakeConversationProvider.calls) == 2
    first_call, second_call = FakeConversationProvider.calls
    assert any(corrected in message.content for message in first_call)
    assert any(corrected in message.content for message in second_call)
    assert any("CBT" in message.content for message in second_call)
    session_id = conversation.output.split("Session ", 1)[1].split(" ·", 1)[0]

    shown = runner.invoke(cli.app, ["sessions", "show", session_id])
    assert shown.exit_code == 0, shown.output
    transcript = " ".join(shown.output.split())
    assert "What helps me feel grounded after work?" in transcript
    assert "That sounds like a useful moment" in transcript

    # A fresh CLI invocation simulates restart; resumed context must preserve
    # both prior turns and the user's correction to the imported memory.
    FakeConversationProvider.calls = []
    resume_prompts = iter(["What helps me feel grounded after work?", "/end"])
    monkeypatch.setattr(cli.typer, "prompt", lambda *_args, **_kwargs: next(resume_prompts))
    resumed = runner.invoke(cli.app, ["session", "resume", session_id])
    assert resumed.exit_code == 0, resumed.output
    assert len(FakeConversationProvider.calls) == 1
    resumed_call = FakeConversationProvider.calls[0]
    assert any(corrected in message.content for message in resumed_call)
    assert any("What helps me feel grounded after work?" in message.content for message in resumed_call)

    forgotten = runner.invoke(cli.app, ["memory", "forget", extracted_id, "--delete-source"])
    assert forgotten.exit_code == 0, forgotten.output
    db = Database(data_dir)
    assert db.get_document(document_id) is None
    assert db.search_memories("grounded workdays") == []
    assert db.get_current_memory_profile() is None
    assert all(row[6] == "deleted" for row in db.list_memories())
    assert db.session_exists(session_id)
    db.close()
