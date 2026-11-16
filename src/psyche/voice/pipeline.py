from __future__ import annotations

import asyncio
import inspect
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from .events import VoiceEvent, VoiceEventType, VoiceState, VoiceStateMachine
from .queues import PlaybackQueue
from .vad import SpeechFrame, SpeechStarted, UtteranceCompleted


@dataclass
class _CapturedUtterance:
    frames: list
    turn_id: str | None = None
    transcription: asyncio.Task | None = None


class VoiceTurnPipeline:
    """Coordinate continuous capture, interruptible turns, and streamed playback."""

    def __init__(self, orchestrator, microphone, vad, transcriber, synthesizer,
                 player, on_event: Callable[[VoiceEvent], object] | None = None,
                 max_audio_chunks: int = 8):
        self.orchestrator = orchestrator
        self.microphone, self.vad = microphone, vad
        self.transcriber, self.synthesizer, self.player = transcriber, synthesizer, player
        self.on_event = on_event
        self.max_audio_chunks = max_audio_chunks
        self.state = VoiceStateMachine()
        self.stage_seconds: dict[str, list[float]] = {"transcription": [], "response": [], "playback": []}
        self._turn_busy = False
        self._barge_event = asyncio.Event()
        self._utterances: asyncio.Queue[_CapturedUtterance | BaseException | None] = asyncio.Queue(maxsize=4)

    async def _emit(self, kind: VoiceEventType, session_id: str, turn_id: str,
                    **payload) -> None:
        if self.on_event is None:
            return
        result = self.on_event(VoiceEvent(type=kind, session_id=session_id,
                                          turn_id=turn_id, payload=payload))
        if inspect.isawaitable(result):
            await result

    async def run(self, session_id: str, stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        self.state.transition(VoiceState.LISTENING)

        if not hasattr(self.vad, "events"):
            try:
                async for utterance in self.vad.utterances(self.microphone.frames()):
                    if stop.is_set():
                        break
                    await self.process_utterance(session_id, utterance)
            finally:
                if self.state.state is VoiceState.ERROR:
                    self.state.transition(VoiceState.IDLE)
                elif self.state.state is not VoiceState.IDLE:
                    self.state.transition(VoiceState.IDLE)
            return

        # A failed run may leave completed utterances queued behind the failure
        # sentinel. They are not replayable after shutdown, so cancel any live
        # transcription tasks and start retries with a clean input queue.
        self._utterances = asyncio.Queue(maxsize=4)

        async def capture() -> None:
            stream_queue: asyncio.Queue | None = None
            turn_id: str | None = None
            transcription: asyncio.Task | None = None
            stream_supported = callable(getattr(self.transcriber, "transcribe_stream", None))

            async def frame_stream(queue: asyncio.Queue):
                while True:
                    frame = await queue.get()
                    if frame is None:
                        break
                    yield frame

            def partial_callback(speech_turn_id: str):
                async def emit_partial(text: str) -> None:
                    await self._emit(VoiceEventType.TRANSCRIPT_PARTIAL, session_id, speech_turn_id, text=text)
                return emit_partial

            try:
                async for event in self.vad.events(self.microphone.frames()):
                    if stop.is_set():
                        break
                    if transcription is not None and transcription.done():
                        # Surface provider failures during capture instead of
                        # waiting for VAD to close the utterance (or shutdown).
                        transcription.result()
                    if isinstance(event, SpeechStarted):
                        turn_id = str(uuid.uuid4())
                        await self._emit(VoiceEventType.SPEECH_STARTED, session_id, turn_id)
                        if stream_supported:
                            stream_queue = asyncio.Queue(maxsize=100)
                            transcription = asyncio.create_task(
                                self.transcriber.transcribe_stream(
                                    frame_stream(stream_queue), on_partial=partial_callback(turn_id)
                                )
                            )
                            # Include pre-roll and onset immediately, before later frames arrive.
                            for frame in event.initial_frames:
                                await stream_queue.put(frame)
                        # A new onset must not cancel the utterance currently
                        # being transcribed. It can still preempt generation or
                        # playback; completed new utterances remain queued.
                        if self.state.state in {VoiceState.THINKING, VoiceState.SPEAKING}:
                            self._barge_event.set()
                    elif isinstance(event, SpeechFrame):
                        if transcription is not None and not transcription.done():
                            await stream_queue.put(event.frame)
                    elif isinstance(event, UtteranceCompleted):
                        if transcription is not None and not transcription.done():
                            await stream_queue.put(None)
                        await self._utterances.put(_CapturedUtterance(
                            frames=event.frames,
                            turn_id=turn_id,
                            transcription=transcription,
                        ))
                        stream_queue, turn_id, transcription = None, None, None
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                await self._utterances.put(exc)
            finally:
                if transcription is not None:
                    if not transcription.done():
                        transcription.cancel()
                    await asyncio.gather(transcription, return_exceptions=True)
                task = asyncio.current_task()
                if task is None or not task.cancelling():
                    await self._utterances.put(None)

        capture_task = asyncio.create_task(capture())
        try:
            while not stop.is_set():
                item = await self._utterances.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                if stop.is_set():
                    break
                if isinstance(item, _CapturedUtterance):
                    await self.process_utterance(
                        session_id, item.frames, turn_id=item.turn_id,
                        transcription=item.transcription,
                    )
                else:
                    await self.process_utterance(session_id, item)
        finally:
            capture_task.cancel()
            await asyncio.gather(capture_task, return_exceptions=True)
            pending_transcriptions = []
            while not self._utterances.empty():
                try:
                    item = self._utterances.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(item, _CapturedUtterance) and item.transcription is not None:
                    if not item.transcription.done():
                        item.transcription.cancel()
                    pending_transcriptions.append(item.transcription)
            if pending_transcriptions:
                await asyncio.gather(*pending_transcriptions, return_exceptions=True)
            if self.state.state is VoiceState.ERROR:
                self.state.transition(VoiceState.IDLE)
            elif self.state.state is not VoiceState.IDLE:
                self.state.transition(VoiceState.IDLE)

    async def _until_barge_in(self, awaitable):
        """Return (completed, result); cancel the operation on new speech onset."""
        operation = asyncio.ensure_future(awaitable)
        interrupt = asyncio.create_task(self._barge_event.wait())
        try:
            done, _ = await asyncio.wait(
                {operation, interrupt}, return_when=asyncio.FIRST_COMPLETED
            )
            if interrupt in done and self._barge_event.is_set():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                return False, None
            interrupt.cancel()
            await asyncio.gather(interrupt, return_exceptions=True)
            return True, await operation
        except BaseException:
            # In particular, Ctrl+C cancels the pipeline coroutine while the
            # streaming transcription task is still running. Always collect
            # both tasks so provider errors do not become "never retrieved".
            for task in (operation, interrupt):
                if not task.done():
                    task.cancel()
            await asyncio.gather(operation, interrupt, return_exceptions=True)
            raise

    async def _return_to_listening(self, session_id: str, turn_id: str, interrupted: bool) -> None:
        if interrupted:
            if self.state.state is VoiceState.SPEAKING:
                self.state.transition(VoiceState.INTERRUPTING)
                self.state.transition(VoiceState.LISTENING)
            elif self.state.state not in {VoiceState.LISTENING, VoiceState.IDLE, VoiceState.ERROR}:
                self.state.transition(VoiceState.LISTENING)
            await self._emit(VoiceEventType.PLAYBACK_INTERRUPTED, session_id, turn_id)
        elif self.state.state is not VoiceState.LISTENING:
            self.state.transition(VoiceState.LISTENING)

    @staticmethod
    def _take_speakable(text: str, final: bool = False) -> tuple[list[str], str]:
        """Split accumulated model text into sentence-sized TTS inputs."""
        chunks: list[str] = []
        while text:
            boundary = re.search(r"(?<=[.!?])\s+|\n+", text)
            if boundary:
                piece = text[:boundary.start()].strip()
                if piece:
                    chunks.append(piece)
                text = text[boundary.end():]
                continue
            if len(text) >= 240:
                cut = text.rfind(" ", 0, 240)
                if cut > 0:
                    chunks.append(text[:cut].strip())
                    text = text[cut + 1:]
                    continue
            break
        if final and text.strip():
            chunks.append(text.strip())
            text = ""
        return chunks, text

    async def process_utterance(self, session_id: str, frames, *, turn_id: str | None = None,
                                transcription: asyncio.Task | None = None) -> str:
        if self.state.state is VoiceState.IDLE:
            self.state.transition(VoiceState.LISTENING)
        speech_started_emitted = turn_id is not None
        turn_id = turn_id or str(uuid.uuid4())
        self._barge_event.clear()
        self._turn_busy = True
        self.state.transition(VoiceState.CAPTURING)
        self.state.transition(VoiceState.TRANSCRIBING)
        started = time.perf_counter()
        if not speech_started_emitted:
            await self._emit(VoiceEventType.SPEECH_STARTED, session_id, turn_id)
        try:
            transcribe = transcription if transcription is not None else self.transcriber.transcribe(frames)
            completed, transcript = await self._until_barge_in(transcribe)
            if not completed:
                await self._return_to_listening(session_id, turn_id, True)
                return ""
            transcript = transcript.strip()
            self.stage_seconds["transcription"].append(time.perf_counter() - started)
            if not transcript:
                await self._return_to_listening(session_id, turn_id, False)
                return ""
            await self._emit(VoiceEventType.TRANSCRIPT_FINAL, session_id, turn_id, text=transcript)
            self.state.transition(VoiceState.THINKING)
            await self._emit(VoiceEventType.GENERATION_STARTED, session_id, turn_id)
            started = time.perf_counter()
            queue = PlaybackQueue(self.max_audio_chunks)
            playback = asyncio.create_task(self.player.play(queue.chunks(), queue.interrupted))
            text_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=3)
            stream_responder = getattr(self.orchestrator, "stream_response_turn", None)
            responder = getattr(self.orchestrator, "respond_turn", None)
            if responder is None and stream_responder is None:
                responder = self.orchestrator.process_turn
            speaking_started = False

            async def enqueue_sentence(sentence: str) -> None:
                nonlocal speaking_started
                if not speaking_started:
                    speaking_started = True
                    self.state.transition(VoiceState.SPEAKING)
                    await self._emit(VoiceEventType.PLAYBACK_STARTED, session_id, turn_id)
                await text_queue.put(sentence)

            async def generate_to_queue() -> str:
                response_parts: list[str] = []
                pending_text = ""

                async def accept_delta(delta: str) -> None:
                    nonlocal pending_text
                    first_delta = not response_parts
                    response_parts.append(delta)
                    pending_text += delta
                    await self._emit(
                        VoiceEventType.RESPONSE_TEXT_DELTA, session_id, turn_id,
                        text=delta, final=False, first=first_delta,
                    )
                    speakable, pending_text = self._take_speakable(pending_text)
                    for sentence in speakable:
                        await enqueue_sentence(sentence)

                if stream_responder is not None:
                    async for delta in stream_responder(session_id, transcript):
                        if delta:
                            await accept_delta(delta)
                else:
                    assert responder is not None
                    response = await responder(session_id, transcript)
                    if response:
                        await accept_delta(response)
                speakable, pending_text = self._take_speakable(pending_text, final=True)
                for sentence in speakable:
                    await enqueue_sentence(sentence)
                await text_queue.put(None)
                return "".join(response_parts).strip()

            async def synthesize_to_queue() -> None:
                while True:
                    sentence = await text_queue.get()
                    if sentence is None:
                        break
                    chunk_count = 0
                    async for chunk in self.synthesizer.synthesize(sentence):
                        await queue.put(chunk)
                        chunk_count += 1
                    if chunk_count:
                        await queue.mark_spoken(sentence)
                await queue.finish()

            generation = asyncio.create_task(generate_to_queue())
            synthesis = asyncio.create_task(synthesize_to_queue())
            interruption = asyncio.create_task(self._barge_event.wait())
            playback_started = time.perf_counter()
            try:
                while not (generation.done() and synthesis.done() and playback.done()):
                    active = {task for task in (generation, synthesis, playback) if not task.done()}
                    done, _ = await asyncio.wait(
                        active | {interruption},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if interruption in done and self._barge_event.is_set():
                        await queue.interrupt()
                        for task in (generation, synthesis, playback):
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(generation, synthesis, playback, return_exceptions=True)
                        self._persist_interrupted_prefix(session_id, queue)
                        await self._return_to_listening(session_id, turn_id, True)
                        return ""
                    for task in done - {interruption}:
                        task.result()
                if self._barge_event.is_set():
                    await queue.interrupt()
                    self._persist_interrupted_prefix(session_id, queue)
                    await self._return_to_listening(session_id, turn_id, True)
                    return ""
                response = generation.result()
                self.stage_seconds["response"].append(time.perf_counter() - started)
                self.stage_seconds["playback"].append(time.perf_counter() - playback_started)
            except BaseException:
                await queue.interrupt()
                for task in (generation, synthesis, playback):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(generation, synthesis, playback, return_exceptions=True)
                self._persist_interrupted_prefix(session_id, queue, "incomplete")
                raise
            finally:
                interruption.cancel()
                await asyncio.gather(interruption, return_exceptions=True)

            persist = getattr(self.orchestrator, "persist_voice_response", None)
            if persist is not None:
                persist(session_id, response)
            await self._emit(VoiceEventType.TURN_COMPLETED, session_id, turn_id)
            await self._return_to_listening(session_id, turn_id, False)
            return response
        except asyncio.CancelledError:
            if self.state.state not in {VoiceState.ERROR, VoiceState.IDLE}:
                self.state.transition(VoiceState.ERROR)
            raise
        except Exception as exc:
            if self.state.state not in {VoiceState.ERROR, VoiceState.IDLE}:
                self.state.transition(VoiceState.ERROR)
            await self._emit(VoiceEventType.ERROR, session_id, turn_id, message=str(exc))
            raise
        finally:
            self._turn_busy = False

    def _persist_interrupted_prefix(self, session_id: str, queue: PlaybackQueue,
                                    delivery_status: str = "interrupted") -> None:
        prefix = " ".join(part.strip() for part in queue.played_text if part.strip()).strip()
        persist_partial = getattr(self.orchestrator, "persist_partial_voice_response", None)
        if prefix and persist_partial is not None:
            persist_partial(session_id, prefix, delivery_status)
            return
        persist = getattr(self.orchestrator, "persist_interrupted_voice_response", None)
        if prefix and persist is not None:
            persist(session_id, prefix)
