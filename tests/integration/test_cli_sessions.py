from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from psyche import cli
from psyche.storage.database import Database


class FakeProvider:
    replies = iter(["A first supportive reply.", "A reply after resuming."])

    def __init__(self, model, timeout):
        pass

    async def generate(self, messages, temperature, max_output_tokens):
        return next(self.replies)


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="Storage uses Windows DPAPI")
def test_cli_session_can_start_leave_open_resume_and_end(tmp_path: Path, monkeypatch):
    config = SimpleNamespace(
        data_dir=tmp_path,
        llm=SimpleNamespace(model="test-model", request_timeout_seconds=2,
                            temperature=0.5, max_output_tokens=100),
        safety=SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(cli, "services", lambda: (config, Database(tmp_path)))
    monkeypatch.setattr(cli, "provider_for", lambda config: FakeProvider(config.model, config.request_timeout_seconds))
    FakeProvider.replies = iter(["A first supportive reply.", "A reply after resuming."])
    inputs = iter(["My first thought.", "EOF", "A later thought.", "/end"])

    def prompt(*args, **kwargs):
        value = next(inputs)
        if value == "EOF":
            raise EOFError
        return value

    monkeypatch.setattr(cli.typer, "prompt", prompt)

    runner = CliRunner()
    started = runner.invoke(cli.app, ["session", "start", "--method", "cbt"])
    assert started.exit_code == 0, started.output
    session_id = started.output.split("Session ", 1)[1].split(" ·", 1)[0]

    # EOF leaves the session active and durable for the next CLI invocation.
    db = Database(tmp_path)
    assert db.get_messages(session_id) == [
        {"role": "user", "content": "My first thought."},
        {"role": "assistant", "content": "A first supportive reply."},
    ]
    assert next(row[4] for row in db.list_sessions() if row[0] == session_id) == "active"
    db.close()
    resumed = runner.invoke(cli.app, ["session", "resume", session_id])
    assert resumed.exit_code == 0, resumed.output
    assert "A reply after resuming." in resumed.output

    db = Database(tmp_path)
    assert db.session_exists(session_id)
    assert [m["content"] for m in db.get_messages(session_id)] == [
        "My first thought.", "A first supportive reply.",
        "A later thought.", "A reply after resuming."
    ]
    status = next(row[4] for row in db.list_sessions() if row[0] == session_id)
    assert status == "completed"
    db.close()


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="Storage uses Windows DPAPI")
def test_cloud_processing_disabled_blocks_start_and_resume(tmp_path: Path, monkeypatch):
    config = SimpleNamespace(
        data_dir=tmp_path,
        privacy=SimpleNamespace(cloud_processing_allowed=False),
        llm=SimpleNamespace(model="test-model", request_timeout_seconds=2),
    )
    database = Database(tmp_path)
    closed = []
    original_close = database.close

    def close():
        closed.append(True)
        original_close()

    database.close = close
    monkeypatch.setattr(cli, "services", lambda: (config, database))
    monkeypatch.setattr(cli, "provider_for", lambda *_args: pytest.fail("provider must not be initialized"))
    runner = CliRunner()

    start = runner.invoke(cli.app, ["session", "start"])
    assert start.exit_code == 2
    assert "Cloud processing is disabled" in start.output
    assert closed == [True]

    # Resume has the same guard and must not invoke the provider either.
    database = Database(tmp_path)
    sid = database.create_session()
    closed.clear()
    original_close = database.close

    def close_again():
        closed.append(True)
        original_close()

    database.close = close_again
    monkeypatch.setattr(cli, "services", lambda: (config, database))
    resume = runner.invoke(cli.app, ["session", "resume", sid])
    assert resume.exit_code == 2
    assert "Cloud processing is disabled" in resume.output
    assert closed == [True]
