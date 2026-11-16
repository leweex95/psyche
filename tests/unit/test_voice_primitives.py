import asyncio

import pytest

from psyche.voice.events import (
    AudioFrame,
    StateTransitionError,
    VoiceState,
    VoiceStateMachine,
)
from psyche.voice.queues import PlaybackQueue
from psyche.voice.vad import UtteranceSegmenter
from psyche.voice.transcription import OpenAIRealtimeTranscriber
from psyche.voice.synthesis import OpenAITTSSynthesizer
from psyche.voice.playback import SoundDeviceAudioPlayer
from psyche.voice.pipeline import VoiceTurnPipeline
import json


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
async def test_realtime_transcriber_sends_audio_and_returns_final_text(monkeypatch):
    class FakeSocket:
        def __init__(self):
            self.sent = []
            self.events = iter([
                {"type": "session.updated"},
                {"type": "conversation.item.input_audio_transcription.delta", "delta": "hello"},
                {"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello there"},
            ])

        async def send(self, message):
            self.sent.append(json.loads(message))

        async def recv(self):
            return json.dumps(next(self.events))

    class FakeConnection:
        def __init__(self):
            self.socket = FakeSocket()

        async def __aenter__(self):
            return self.socket

        async def __aexit__(self, *_args):
            return False

    connection = FakeConnection()
    calls = []

    def connect(url, **kwargs):
        calls.append((url, kwargs))
        return connection

    monkeypatch.setenv("PSYCHE_LLM_API_KEY", "test-key")
    partials = []
    transcriber = OpenAIRealtimeTranscriber(connector=connect, on_partial=partials.append)
    result = await transcriber.transcribe([
        AudioFrame(data=b"\x01\x00", sample_rate=16000, captured_at=0),
    ])

    assert result == "hello there"
    assert partials == ["hello"]
    assert calls[0][0].endswith("?intent=transcription")
    assert connection.socket.sent[0]["type"] == "session.update"
    assert connection.socket.sent[1]["type"] == "input_audio_buffer.append"
    assert connection.socket.sent[2]["type"] == "input_audio_buffer.commit"


@pytest.mark.asyncio
async def test_realtime_transcriber_emits_partial_before_capture_finishes(monkeypatch):
    partial_ready = asyncio.Event()
    release_capture = asyncio.Event()

    class FakeSocket:
        def __init__(self):
            self.audio_sent = asyncio.Event()
            self.committed = asyncio.Event()
            self.receives = 0

        async def send(self, message):
            event = json.loads(message)
            if event["type"] == "input_audio_buffer.append":
                self.audio_sent.set()
            elif event["type"] == "input_audio_buffer.commit":
                self.committed.set()

        async def recv(self):
            self.receives += 1
            if self.receives == 1:
                return json.dumps({"type": "session.updated"})
            if self.receives == 2:
                await self.audio_sent.wait()
                return json.dumps({"type": "conversation.item.input_audio_transcription.delta", "delta": "hello"})
            await self.committed.wait()
            return json.dumps({"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello there"})

    class Connection:
        def __init__(self):
            self.socket = FakeSocket()

        async def __aenter__(self):
            return self.socket

        async def __aexit__(self, *_args):
            return False

    connection = Connection()
    monkeypatch.setenv("PSYCHE_LLM_API_KEY", "test-key")

    async def frames():
        yield AudioFrame(data=b"\x01\x00", sample_rate=16000, captured_at=0)
        await release_capture.wait()

    partials = []

    def on_partial(text):
        partials.append(text)
        partial_ready.set()

    transcriber = OpenAIRealtimeTranscriber(connector=lambda *_args, **_kwargs: connection)
    task = asyncio.create_task(transcriber.transcribe_stream(frames(), on_partial=on_partial))
    await asyncio.wait_for(partial_ready.wait(), timeout=1)
    assert not release_capture.is_set()
    assert partials == ["hello"]
    release_capture.set()
    assert await asyncio.wait_for(task, timeout=1) == "hello there"


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
async def test_tts_streams_pcm_chunks_with_configured_voice():
    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def iter_bytes(self, chunk_size):
            assert chunk_size == 4096
            yield b"\x01\x00"
            yield b"\x02\x00"

    class Create:
        def __init__(self):
            self.args = None

        def __call__(self, **kwargs):
            self.args = kwargs
            return Response()

    create = Create()
    client = type("Client", (), {})()
    client.audio = type("Audio", (), {})()
    client.audio.speech = type("Speech", (), {})()
    client.audio.speech.with_streaming_response = type("Streaming", (), {})()
    client.audio.speech.with_streaming_response.create = create

    synthesizer = OpenAITTSSynthesizer("test-model", "test-voice", client=client)
    chunks = [chunk async for chunk in synthesizer.synthesize("Hello there.")]
    assert chunks == [b"\x01\x00", b"\x02\x00"]
    assert create.args == {
        "model": "test-model", "voice": "test-voice",
        "input": "Hello there.", "response_format": "pcm",
    }


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
