import asyncio
import struct
import sys

import pytest

from psyche.voice.events import (
    AudioFrame,
    StateTransitionError,
    VoiceState,
    VoiceStateMachine,
)
from psyche.voice.queues import PlaybackQueue
from psyche.voice.vad import SpeechFrame, SpeechStarted, UtteranceCompleted, UtteranceSegmenter
from psyche.voice.transcription import LocalWhisperTranscriber, TranscriptionError
from psyche.voice.synthesis import LocalPiperSynthesizer
from psyche.voice.playback import SoundDeviceAudioPlayer
from psyche.voice.pipeline import VoiceTurnPipeline
from psyche.voice.audio import SoundDeviceMicrophone
from psyche.voice.diagnostics import capture_input_levels


@pytest.mark.asyncio
async def test_pipeline_cancellation_collects_failed_transcription_task():
    pipeline = VoiceTurnPipeline(None, None, None, None, None, None)
    operation = asyncio.get_running_loop().create_future()
    waiter = asyncio.create_task(pipeline._until_barge_in(operation))
    await asyncio.sleep(0)

    operation.set_exception(TranscriptionError("provider quota exhausted"))
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert not operation._log_traceback
    assert all(task.done() for task in asyncio.all_tasks() if task is not asyncio.current_task())


@pytest.mark.asyncio
async def test_pipeline_surfaces_streaming_transcription_failure_during_capture():
    frame = AudioFrame(data=b"\x01\x00", sample_rate=16000, captured_at=0)

    class Microphone:
        async def frames(self):
            yield frame

    class VAD:
        async def events(self, _frames):
            yield SpeechStarted([frame])
            await asyncio.sleep(0)
            yield SpeechFrame(frame)

    class Transcriber:
        async def transcribe_stream(self, _frames, on_partial=None):
            raise TranscriptionError("provider account has no credits")

    pipeline = VoiceTurnPipeline(None, Microphone(), VAD(), Transcriber(), None, None)

    with pytest.raises(TranscriptionError, match="no credits"):
        await asyncio.wait_for(pipeline.run("session-1"), timeout=1)

    assert pipeline.state.state is VoiceState.IDLE


def test_voice_state_machine_accepts_plan_flow_and_rejects_invalid_edges():
    machine = VoiceStateMachine()
    flow = [
        VoiceState.LISTENING,
        VoiceState.CAPTURING,
        VoiceState.TRANSCRIBING,
        VoiceState.THINKING,
        VoiceState.SPEAKING,
        VoiceState.INTERRUPTING,
        VoiceState.LISTENING,
    ]
    for state in flow:
        machine.transition(state)
    assert machine.state is VoiceState.LISTENING
    with pytest.raises(StateTransitionError):
        machine.transition(VoiceState.SPEAKING)


def test_audio_frame_is_typed_and_rejects_invalid_sample_rate():
    frame = AudioFrame(data=b"\0\0", sample_rate=16000, captured_at=1.0)
    assert frame.channels == 1
    with pytest.raises(ValueError):
        AudioFrame(data=b"", sample_rate=0, captured_at=0)


@pytest.mark.asyncio
async def test_microphone_negotiates_array_format_downmixes_and_resamples(monkeypatch):
    settings = []

    class FakeStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            # Four distinct channels at 48 kHz; the expected mono sample is 250.
            samples = [100, 200, 300, 400] * 1440
            pcm = b"".join(struct.pack("<h", sample) for sample in samples)
            self.kwargs["callback"](pcm, 1440, None, 0)
            return self

        def __exit__(self, *_args):
            return False

    class FakeSoundDevice:
        @staticmethod
        def query_devices(_device, _kind):
            return {"max_input_channels": 4, "default_samplerate": 48000}

        @staticmethod
        def check_input_settings(**kwargs):
            settings.append((kwargs["samplerate"], kwargs["channels"]))
            if (kwargs["samplerate"], kwargs["channels"]) != (48000, 4):
                raise ValueError("unsupported input format")

        RawInputStream = FakeStream

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)
    microphone = SoundDeviceMicrophone(device=15)
    frames = microphone.frames()
    frame = await anext(frames)
    await frames.aclose()

    assert settings == [(16000, 4), (16000, 1), (48000, 4)]
    assert frame.sample_rate == 16000
    assert len(frame.data) == 960
    assert set(struct.iter_unpack("<h", frame.data)) == {(250,)}


def test_microphone_diagnostic_reports_channels_and_app_mix_without_returning_audio():
    class FakeStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            rows = [(1000, -1000)] * 80 + [(1000, 1000)] * 80
            pcm = b"".join(struct.pack("<hh", *row) for row in rows)
            self.kwargs["callback"](pcm, len(rows), None, 0)
            return self

        def __exit__(self, *_args):
            return False

    class FakeSoundDevice:
        RawInputStream = FakeStream

        @staticmethod
        def query_devices(_device, _kind):
            return {
                "index": 0, "name": "Test microphone", "hostapi": 0,
                "max_input_channels": 2, "default_samplerate": 16000,
            }

        @staticmethod
        def check_input_settings(**_kwargs):
            return None

    levels = capture_input_levels(FakeSoundDevice, 0, target_rate=16000, seconds=0.01)

    assert levels["sample_rate"] == 16000
    assert levels["channels"] == 2
    assert levels["frames"] == 160
    assert levels["channel_levels"][0]["rms"] == 1000
    assert levels["channel_levels"][1]["rms"] == 1000
    assert levels["mixed_level"]["peak"] == 1000
    assert levels["mixed_level"]["nonzero"] == 80
    assert not any(isinstance(value, bytes) for value in levels.values())


@pytest.mark.asyncio
async def test_playback_queue_preserves_order_and_ends_cleanly():
    queue = PlaybackQueue(max_chunks=2)
    async def produce():
        await queue.put(b"a")
        await queue.put(b"b")
        await queue.finish()

    producer = asyncio.create_task(produce())
    assert [chunk async for chunk in queue.chunks()] == [b"a", b"b"]
    await producer


@pytest.mark.asyncio
async def test_playback_interruption_discards_unplayed_audio():
    queue = PlaybackQueue(max_chunks=2)
    await queue.put(b"stale")
    await queue.interrupt()
    assert [chunk async for chunk in queue.chunks()] == []
    assert queue.interrupted.is_set()


@pytest.mark.asyncio
async def test_playback_queue_marks_only_completed_sentence_audio_as_played():
    queue = PlaybackQueue(max_chunks=3)
    await queue.put(b"pcm")
    await queue.mark_spoken("A complete sentence.")
    await queue.put(b"unfinished")
    iterator = queue.chunks()
    assert await anext(iterator) == b"pcm"
    # Requesting the next item acknowledges the prior device write and consumes
    # the sentence marker, but not the following unfinished audio.
    assert await anext(iterator) == b"unfinished"
    assert queue.played_text == ["A complete sentence."]
    await iterator.aclose()


@pytest.mark.asyncio
async def test_vad_segments_speech_and_includes_pre_roll():
    class FakeClassifier:
        def is_speech(self, pcm, sample_rate):
            return pcm == b"speech"

    async def frames():
        for index, data in enumerate([b"quiet", b"speech", b"speech", b"quiet", b"quiet"]):
            yield AudioFrame(data=data, sample_rate=16000, captured_at=float(index))

    segmenter = UtteranceSegmenter(
        FakeClassifier(), sample_rate=16000, frame_ms=30,
        end_silence_ms=60, pre_roll_ms=30, min_speech_ms=60,
    )
    utterances = [u async for u in segmenter.utterances(frames())]
    assert len(utterances) == 1
    assert [frame.data for frame in utterances[0]] == [b"quiet", b"speech", b"speech", b"quiet", b"quiet"]


@pytest.mark.asyncio
async def test_vad_discards_short_false_trigger():
    class FakeClassifier:
        def is_speech(self, pcm, sample_rate):
            return pcm == b"speech"

    async def frames():
        for index, data in enumerate([b"speech", b"quiet", b"quiet"]):
            yield AudioFrame(data=data, sample_rate=16000, captured_at=float(index))

    segmenter = UtteranceSegmenter(
        FakeClassifier(), frame_ms=30, end_silence_ms=60, min_speech_ms=60,
    )
    assert [u async for u in segmenter.utterances(frames())] == []


@pytest.mark.asyncio
async def test_local_whisper_transcriber_converts_pcm_and_returns_transcript():
    import numpy as np

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            assert audio.dtype == np.float32
            assert np.allclose(audio, [0.5, -0.5])
            assert kwargs["vad_filter"] is False
            return iter([type("Segment", (), {"text": " local words "})()]), object()

    transcriber = LocalWhisperTranscriber(model_factory=lambda *_args, **_kwargs: FakeModel())
    result = await transcriber.transcribe([
        AudioFrame(data=struct.pack("<hh", 16384, -16384), sample_rate=16000, captured_at=0),
    ])
    assert result == "local words"


@pytest.mark.asyncio
async def test_local_whisper_transcriber_rejects_non_mono_or_wrong_rate():
    transcriber = LocalWhisperTranscriber(model_factory=lambda *_args, **_kwargs: pytest.fail("must not load"))
    with pytest.raises(TranscriptionError, match="16 kHz"):
        await transcriber.transcribe([AudioFrame(data=b"\0\0", sample_rate=24000, captured_at=0)])


@pytest.mark.asyncio
async def test_local_whisper_file_transcription_uses_local_model(tmp_path):
    class FakeModel:
        def transcribe(self, path, **kwargs):
            assert path.endswith("sample.wav")
            assert kwargs["vad_filter"] is True
            return iter([type("Segment", (), {"text": " file transcript "})()]), object()

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"not decoded by fake")
    transcriber = LocalWhisperTranscriber(model_factory=lambda *_args, **_kwargs: FakeModel())
    assert await transcriber.transcribe_file(audio_file) == "file transcript"


@pytest.mark.asyncio
async def test_non_streaming_pipeline_emits_speech_started_once_per_utterance():
    events = []
    frame = AudioFrame(data=b"speech", sample_rate=16000, captured_at=0)

    class Microphone:
        async def frames(self):
            yield frame

    class VAD:
        async def events(self, _frames):
            yield SpeechStarted([frame])
            yield SpeechFrame(frame)
            yield UtteranceCompleted([frame])

    class Transcriber:
        async def transcribe(self, _frames):
            return "Local test transcript."

    class Orchestrator:
        async def respond_turn(self, _session_id, _text):
            return "Hello."

    class Synthesizer:
        async def synthesize(self, _text):
            yield b"\x01\x00"

    class Player:
        async def play(self, chunks, _interrupted):
            _ = [chunk async for chunk in chunks]

    pipeline = VoiceTurnPipeline(
        Orchestrator(), Microphone(), VAD(), Transcriber(), Synthesizer(), Player(),
        on_event=events.append,
    )
    await asyncio.wait_for(pipeline.run("session-1"), timeout=1)

    event_types = [event.type.value for event in events]
    assert event_types.count("speech_started") == 1
    assert "transcript_final" in event_types
    assert "turn_completed" in event_types


@pytest.mark.asyncio
async def test_pipeline_emits_partial_transcript_during_vad_capture():
    events = []

    class Microphone:
        async def frames(self):
            for index, data in enumerate((b"speech", b"quiet")):
                yield AudioFrame(data=data, sample_rate=16000, captured_at=float(index))

    class Classifier:
        def is_speech(self, pcm, _sample_rate):
            return pcm == b"speech"

    class Transcriber:
        async def transcribe(self, _frames):
            raise AssertionError("incremental adapter should be used during capture")

        async def transcribe_stream(self, frames, on_partial=None):
            received = []
            async for frame in frames:
                received.append(frame.data)
                if len(received) == 1:
                    await on_partial("Hello")
            assert received == [b"speech", b"quiet"]
            return "Hello there."

    class Orchestrator:
        async def stream_response_turn(self, _session_id, _text):
            yield "Hi there."

    class Synthesizer:
        async def synthesize(self, _text):
            yield b"\x01\x00"

    class Player:
        async def play(self, chunks, _interrupted):
            self.played = [chunk async for chunk in chunks]

    vad = UtteranceSegmenter(
        Classifier(), frame_ms=30, end_silence_ms=30, pre_roll_ms=0, min_speech_ms=30,
    )
    pipeline = VoiceTurnPipeline(
        Orchestrator(), Microphone(), vad, Transcriber(), Synthesizer(), Player(),
        on_event=events.append,
    )
    await asyncio.wait_for(pipeline.run("session-1"), timeout=1)

    event_types = [event.type.value for event in events]
    assert "transcript_partial" in event_types
    assert "transcript_final" in event_types
    assert event_types.index("transcript_partial") < event_types.index("transcript_final")
    assert next(event.payload["text"] for event in events if event.type.value == "transcript_partial") == "Hello"


@pytest.mark.asyncio
async def test_piper_synthesizer_generates_local_pcm_in_bounded_chunks(tmp_path):
    class FakeVoice:
        def synthesize(self, text):
            assert text == "Hello there."
            yield type("AudioChunk", (), {
                "sample_rate": 24000,
                "sample_width": 2,
                "sample_channels": 1,
                "audio_int16_bytes": bytes(5000),
            })()

    synthesizer = LocalPiperSynthesizer(
        "test-voice", tmp_path, voice_loader=lambda *_args, **_kwargs: FakeVoice(),
        downloader=lambda *_args: None,
    )
    chunks = [chunk async for chunk in synthesizer.synthesize("Hello there.")]
    assert [len(chunk) for chunk in chunks] == [4096, 904]
    assert b"".join(chunks) == bytes(5000)


@pytest.mark.asyncio
async def test_player_writes_streamed_chunks_and_honors_interrupt():
    class FakeStream:
        def __init__(self, **kwargs):
            self.writes = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, chunk):
            self.writes.append(chunk)

    streams = []

    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    async def chunks():
        yield b"\x01\x00"
        yield b"\x02\x00"

    interrupted = asyncio.Event()
    await SoundDeviceAudioPlayer(stream_factory=factory).play(chunks(), interrupted)
    assert streams[0].writes == [b"\x01\x00", b"\x02\x00"]

    interrupted.set()
    await SoundDeviceAudioPlayer(stream_factory=factory).play(chunks(), interrupted)
    assert streams[1].writes == []


@pytest.mark.asyncio
async def test_player_resamples_for_selected_device_native_rate(monkeypatch):
    class FakeStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.writes = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, chunk):
            self.writes.append(chunk)

    streams = []
    checks = []

    class FakeSoundDevice:
        @staticmethod
        def make_stream(**kwargs):
            stream = FakeStream(**kwargs)
            streams.append(stream)
            return stream

        RawOutputStream = staticmethod(make_stream)

        @staticmethod
        def check_output_settings(**kwargs):
            checks.append(kwargs["samplerate"])
            if kwargs["samplerate"] == 24000:
                raise ValueError("device does not support this rate")

        @staticmethod
        def query_devices(_device, _kind):
            return {"default_samplerate": 44100}

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)

    async def chunks():
        yield bytes(640)
        yield bytes(640)

    await SoundDeviceAudioPlayer(device=17).play(chunks(), asyncio.Event())

    assert checks == [24000, 44100]
    assert streams[0].kwargs["samplerate"] == 44100
    assert [len(chunk) for chunk in streams[0].writes] == [1174, 1176]


@pytest.mark.asyncio
async def test_voice_pipeline_persists_complete_turn_after_streamed_playback():
    events = []

    class Orchestrator:
        def __init__(self):
            self.turns = []

        async def process_turn(self, session_id, text):
            self.turns.append((session_id, text))
            return "A considered reply."

    class Transcriber:
        async def transcribe(self, frames):
            return "I feel overwhelmed."

    class Synthesizer:
        async def synthesize(self, text):
            assert text == "A considered reply."
            yield b"\x01\x00"
            yield b"\x02\x00"

    class Player:
        async def play(self, chunks, interrupted):
            self.played = [chunk async for chunk in chunks]

    player, orchestrator = Player(), Orchestrator()
    pipeline = VoiceTurnPipeline(orchestrator, None, None, Transcriber(), Synthesizer(), player,
                                 on_event=events.append)
    result = await pipeline.process_utterance("session-1", [object()])
    assert result == "A considered reply."
    assert orchestrator.turns == [("session-1", "I feel overwhelmed.")]
    assert player.played == [b"\x01\x00", b"\x02\x00"]
    assert pipeline.state.state is VoiceState.LISTENING
    assert [event.type for event in events] == [
        "speech_started", "transcript_final", "generation_started", "response_text_delta",
        "playback_started", "turn_completed",
    ]
    assert pipeline.stage_seconds["transcription"]
    assert pipeline.stage_seconds["response"]
    assert pipeline.stage_seconds["playback"]


@pytest.mark.asyncio
async def test_voice_pipeline_starts_tts_before_llm_stream_finishes():
    tts_started = asyncio.Event()
    events = []

    class Orchestrator:
        def __init__(self):
            self.stream_finished = False
            self.delivered = []

        async def stream_response_turn(self, _session_id, _text):
            yield "The first sentence is ready. "
            await tts_started.wait()
            assert not self.stream_finished
            self.stream_finished = True
            yield "Here is the second sentence."

        def persist_voice_response(self, session_id, response):
            self.delivered.append((session_id, response))

    class Transcriber:
        async def transcribe(self, _frames):
            return "Can you help?"

    class Synthesizer:
        def __init__(self):
            self.inputs = []

        async def synthesize(self, text):
            self.inputs.append(text)
            tts_started.set()
            yield b"\x01\x00"

    class Player:
        async def play(self, chunks, interrupted):
            self.played = [chunk async for chunk in chunks]

    orchestrator = Orchestrator()
    synthesizer = Synthesizer()
    player = Player()
    pipeline = VoiceTurnPipeline(
        orchestrator, None, None, Transcriber(), synthesizer, player, on_event=events.append
    )

    result = await asyncio.wait_for(pipeline.process_utterance("session-1", [object()]), timeout=1)

    assert synthesizer.inputs == ["The first sentence is ready.", "Here is the second sentence."]
    assert player.played == [b"\x01\x00", b"\x01\x00"]
    assert result == "The first sentence is ready. Here is the second sentence."
    assert orchestrator.delivered == [("session-1", result)]
    assert [event.payload["text"] for event in events if event.type.value == "response_text_delta"] == [
        "The first sentence is ready. ", "Here is the second sentence."
    ]


@pytest.mark.asyncio
async def test_voice_barge_in_cancels_speech_and_does_not_mark_reply_delivered():
    class Orchestrator:
        def __init__(self):
            self.delivered = []

        async def respond_turn(self, _session_id, _text):
            return "A response that must not be recorded as delivered."

        def persist_voice_response(self, session_id, response):
            self.delivered.append((session_id, response))

    class Transcriber:
        async def transcribe(self, _frames):
            return "Please continue."

    class Synthesizer:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        async def synthesize(self, _text):
            yield b"\x01\x00"
            self.pipeline._barge_event.set()
            await asyncio.sleep(0)
            yield b"\x02\x00"

    class Player:
        async def play(self, chunks, interrupted):
            async for _ in chunks:
                await asyncio.sleep(0)

    orchestrator = Orchestrator()
    pipeline = VoiceTurnPipeline(orchestrator, None, None, Transcriber(), None, Player())
    pipeline.synthesizer = Synthesizer(pipeline)

    result = await asyncio.wait_for(pipeline.process_utterance("session-1", [object()]), timeout=1)

    assert result == ""
    assert orchestrator.delivered == []
    assert pipeline.state.state is VoiceState.LISTENING


@pytest.mark.asyncio
async def test_voice_barge_in_persists_only_fully_played_sentence_prefix(monkeypatch):
    first_sentence_played = asyncio.Event()
    queues = []
    original_queue = PlaybackQueue

    def capture_queue(*args, **kwargs):
        queue = original_queue(*args, **kwargs)
        queues.append(queue)
        return queue

    monkeypatch.setattr("psyche.voice.pipeline.PlaybackQueue", capture_queue)

    class Orchestrator:
        def __init__(self):
            self.interrupted = []

        async def stream_response_turn(self, _session_id, _text):
            yield "First sentence. "
            yield "Second sentence is interrupted."

        def persist_interrupted_voice_response(self, session_id, text):
            self.interrupted.append((session_id, text))

    class Transcriber:
        async def transcribe(self, _frames):
            return "Continue."

    class Synthesizer:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        async def synthesize(self, text):
            if text.startswith("Second"):
                await first_sentence_played.wait()
                self.pipeline._barge_event.set()
            yield b"\x01\x00"

    class Player:
        async def play(self, chunks, interrupted):
            iterator = chunks.__aiter__()
            await iterator.__anext__()
            # Pulling once more consumes the sentence marker only after the
            # previous audio write, then blocks waiting for second-sentence audio.
            next_chunk = asyncio.create_task(iterator.__anext__())
            while not queues[0].played_text:
                await asyncio.sleep(0)
            first_sentence_played.set()
            try:
                await next_chunk
            except StopAsyncIteration:
                pass

    orchestrator = Orchestrator()
    pipeline = VoiceTurnPipeline(orchestrator, None, None, Transcriber(), None, Player())
    pipeline.synthesizer = Synthesizer(pipeline)
    result = await asyncio.wait_for(pipeline.process_utterance("session-1", [object()]), timeout=1)

    assert result == ""
    assert orchestrator.interrupted == [("session-1", "First sentence.")]


@pytest.mark.asyncio
async def test_voice_failure_preserves_only_played_prefix_as_incomplete(monkeypatch):
    first_sentence_played = asyncio.Event()
    queues = []
    original_queue = PlaybackQueue

    def capture_queue(*args, **kwargs):
        queue = original_queue(*args, **kwargs)
        queues.append(queue)
        return queue

    monkeypatch.setattr("psyche.voice.pipeline.PlaybackQueue", capture_queue)

    class Orchestrator:
        def __init__(self):
            self.partial = []

        async def stream_response_turn(self, _session_id, _text):
            yield "First sentence. "
            yield "Second sentence."

        def persist_partial_voice_response(self, session_id, text, status):
            self.partial.append((session_id, text, status))

    class Transcriber:
        async def transcribe(self, _frames):
            return "Continue."

    class Synthesizer:
        async def synthesize(self, text):
            if text.startswith("Second"):
                await first_sentence_played.wait()
                raise RuntimeError("synthetic TTS failure")
            yield b"\x01\x00"

    class Player:
        async def play(self, chunks, _interrupted):
            iterator = chunks.__aiter__()
            await iterator.__anext__()
            next_chunk = asyncio.create_task(iterator.__anext__())
            try:
                while not queues[0].played_text:
                    await asyncio.sleep(0)
                first_sentence_played.set()
                await next_chunk
            except StopAsyncIteration:
                pass

    orchestrator = Orchestrator()
    pipeline = VoiceTurnPipeline(orchestrator, None, None, Transcriber(), Synthesizer(), Player())

    with pytest.raises(RuntimeError, match="synthetic TTS failure"):
        await asyncio.wait_for(pipeline.process_utterance("session-1", [object()]), timeout=1)

    assert orchestrator.partial == [("session-1", "First sentence.", "incomplete")]


@pytest.mark.asyncio
async def test_voice_pipeline_keeps_turn_if_tts_fails_and_enters_error_state():
    class Orchestrator:
        async def process_turn(self, _session_id, _text):
            return "Persisted text reply"

    class Transcriber:
        async def transcribe(self, _frames):
            return "Please help me think."

    class Synthesizer:
        async def synthesize(self, _text):
            raise RuntimeError("provider unavailable")
            yield b""

    class Player:
        async def play(self, chunks, interrupted):
            async for _ in chunks:
                pass

    pipeline = VoiceTurnPipeline(Orchestrator(), None, None, Transcriber(), Synthesizer(), Player())
    with pytest.raises(RuntimeError):
        await pipeline.process_utterance("session-1", [object()])
    assert pipeline.state.state is VoiceState.ERROR


@pytest.mark.asyncio
async def test_voice_pipeline_does_not_hang_if_audio_device_fails():
    class Orchestrator:
        async def process_turn(self, _session_id, _text):
            return "A long response."

    class Transcriber:
        async def transcribe(self, _frames):
            return "Tell me more."

    class Synthesizer:
        async def synthesize(self, _text):
            for _ in range(100):
                yield b"\x00\x00"

    class Player:
        async def play(self, chunks, interrupted):
            raise OSError("device disconnected")

    pipeline = VoiceTurnPipeline(Orchestrator(), None, None, Transcriber(), Synthesizer(), Player(),
                                 max_audio_chunks=1)
    with pytest.raises(OSError):
        await asyncio.wait_for(pipeline.process_utterance("session-1", [object()]), timeout=1)
