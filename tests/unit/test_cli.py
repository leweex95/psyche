from types import SimpleNamespace
import struct
import sys

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


def test_audio_check_is_local_and_reports_selected_input_levels(monkeypatch):
    info = {
        "index": 0, "name": "Test microphone", "hostapi": 0,
        "max_input_channels": 1, "default_samplerate": 16000,
    }

    class FakeStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            pcm = struct.pack("<h", 500) * 16000
            self.kwargs["callback"](pcm, 16000, None, 0)
            return self

        def __exit__(self, *_args):
            return False

    class FakeSoundDevice:
        RawInputStream = FakeStream
        default = SimpleNamespace(device=(0, -1))

        @staticmethod
        def query_devices(device=None, kind=None):
            return info if kind == "input" else [info]

        @staticmethod
        def query_hostapis():
            return ({"name": "Test API"},)

        @staticmethod
        def check_input_settings(**_kwargs):
            return None

    config = SimpleNamespace(voice=SimpleNamespace(input_device=None, sample_rate=16000))
    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)
    monkeypatch.setattr(cli, "load_config", lambda: config)

    result = CliRunner().invoke(cli.app, ["audio", "check", "--seconds", "1"])

    assert result.exit_code == 0, result.output
    assert "Test microphone" in result.output
    assert "Psyche mono mix: RMS 500.0" in result.output
    assert "not saved" in result.output
    assert "uploaded" in result.output
