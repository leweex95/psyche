from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


class PlaybackQueue:
    """Bounded audio queue with prompt cancellation for barge-in."""

    _END = object()

    def __init__(self, max_chunks: int = 8) -> None:
        if max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        self._queue: asyncio.Queue[bytes | _SpokenMarker | object] = asyncio.Queue(maxsize=max_chunks)
        self.interrupted = asyncio.Event()
        self.played_text: list[str] = []

    async def put(self, chunk: bytes) -> None:
        if self.interrupted.is_set():
            return
        await self._queue.put(chunk)

    async def finish(self) -> None:
        if not self.interrupted.is_set():
            await self._queue.put(self._END)

    async def mark_spoken(self, text: str) -> None:
        """Mark a sentence after its final audio chunk has been enqueued."""
        if text and not self.interrupted.is_set():
            await self._queue.put(_SpokenMarker(text))

    async def interrupt(self) -> None:
        self.interrupted.set()
        # Wake a consumer waiting for the next chunk; discard queued audio so
        # stale speech cannot play after the user has interrupted.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        try:
            self._queue.put_nowait(self._END)
        except asyncio.QueueFull:
            pass

    async def chunks(self) -> AsyncIterator[bytes]:
        while True:
            get_task = asyncio.create_task(self._queue.get())
            interrupt_task = asyncio.create_task(self.interrupted.wait())
            done, pending = await asyncio.wait(
                {get_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if interrupt_task in done and self.interrupted.is_set():
                return
            item = get_task.result()
            if item is self._END:
                return
            if isinstance(item, _SpokenMarker):
                # The consumer requests the next item only after finishing its
                # current write, so reaching this marker confirms the complete
                # preceding sentence audio was written to the output device.
                self.played_text.append(item.text)
                continue
            yield item  # type: ignore[misc]


class _SpokenMarker:
    def __init__(self, text: str):
        self.text = text
