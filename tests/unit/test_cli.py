from types import SimpleNamespace

from typer.testing import CliRunner

from psyche import cli


class FakeDB:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1

    def session_exists(self, _session_id):
        return True


def test_provider_initialization_failure_closes_database_for_start_and_resume(monkeypatch):
    config = SimpleNamespace(
        privacy=SimpleNamespace(cloud_processing_allowed=True),
        llm=SimpleNamespace(model="test-model", request_timeout_seconds=1),
    )
    database = FakeDB()
    monkeypatch.setattr(cli, "services", lambda: (config, database))

    def fail_provider(*_args):
        raise cli.ProviderError("safe provider failure")

    monkeypatch.setattr(cli, "OpenAIProvider", fail_provider)
    runner = CliRunner()

    start = runner.invoke(cli.app, ["session", "start"])
    assert start.exit_code == 2
    assert "safe provider failure" in start.output
    assert database.closed == 1

    resume = runner.invoke(cli.app, ["session", "resume", "existing-session"])
    assert resume.exit_code == 2
    assert "safe provider failure" in resume.output
    assert database.closed == 2
