from types import SimpleNamespace
import struct
import sys
from pathlib import Path

import pytest
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

    monkeypatch.setattr(cli, "provider_for", fail_provider)
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


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("A clear local test transcript.", "Local Whisper transcript: A clear local test transcript."),
        ("", "before VAD and before any LLM request"),
    ],
)
def test_audio_transcribe_check_uses_only_local_capture_and_whisper(monkeypatch, transcript, expected):
    config = SimpleNamespace(
        voice=SimpleNamespace(input_device=None, sample_rate=16000, frame_ms=30, stt_model="base"),
        data_dir=Path("test-data"),
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)
    constructions = []

    class FakeTranscriber:
        def __init__(self, *args):
            constructions.append(("transcriber", args))

        def prepare(self):
            constructions.append(("prepare",))

        async def transcribe(self, frames):
            constructions.append(("transcribe", len(frames)))
            return transcript

    class FakeMicrophone:
        def __init__(self, **kwargs):
            constructions.append(("microphone", kwargs))

        async def frames(self):
            frame = SimpleNamespace(data=b"\0\0" * 16000)
            yield frame
            yield frame

    monkeypatch.setattr(cli, "LocalWhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "SoundDeviceMicrophone", FakeMicrophone)
    monkeypatch.setattr(cli, "provider_for", lambda *_args: pytest.fail("local audio check must not initialize an LLM"))

    result = CliRunner().invoke(cli.app, ["audio", "transcribe-check", "--seconds", "2"])

    assert result.exit_code == 0, result.output
    assert expected in result.output
    assert "not saved, uploaded, or sent to an LLM" in result.output
    assert ("transcribe", 2) in constructions


def test_import_audio_uses_local_transcriber_without_cloud_provider(tmp_path, monkeypatch):
    audio_file = tmp_path / "speech.wav"
    audio_file.write_bytes(b"synthetic local audio")

    class FakeDB:
        def __init__(self):
            self.stored = None
            self.closed = False

        def has_document_hash(self, _digest):
            return False

        def import_document(self, name, text, digest):
            self.stored = (name, text, digest)
            return "doc-local"

        def add_memory_with_provenance(self, *_args, **_kwargs):
            return "memory", True, True

        def close(self):
            self.closed = True

    database = FakeDB()
    config = SimpleNamespace(
        data_dir=tmp_path / "data", voice=SimpleNamespace(stt_model="tiny"),
        privacy=SimpleNamespace(cloud_processing_allowed=False),
    )
    monkeypatch.setattr(cli, "services", lambda: (config, database))
    calls = []

    class FakeTranscriber:
        def __init__(self, model, model_dir):
            calls.append((model, model_dir))

        async def transcribe_file(self, path):
            assert path == audio_file
            return "A local transcript long enough for a candidate."

    monkeypatch.setattr(cli, "LocalWhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "provider_for", lambda *_args: pytest.fail("audio import must not initialize an LLM"))

    result = CliRunner().invoke(cli.app, ["import-audio", str(audio_file)], input="y\n")

    assert result.exit_code == 0, result.output
    assert "locally" in result.output
    assert calls == [("tiny", config.data_dir / "models" / "faster-whisper")]
    assert database.stored[1] == "A local transcript long enough for a candidate."
    assert database.closed
