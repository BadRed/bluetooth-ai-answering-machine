"""Deepgram Flux streaming speech-to-text with built-in end-of-turn detection."""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode

import websockets

log = logging.getLogger("answering_machine.stt")

URL = "wss://api.deepgram.com/v2/listen"
CHUNK_SECONDS = 0.08  # Flux works best with ~80 ms audio chunks


@dataclass
class TurnEvent:
    event: str  # StartOfTurn | Update | EagerEndOfTurn | TurnResumed | EndOfTurn
    transcript: str
    turn_index: int


class FluxSTT:
    def __init__(self, api_key: str, sample_rate: int = 8000, keyterms: list[str] | None = None,
                 eager_eot_threshold: float | None = 0.5):
        params = [("model", "flux-general-en"), ("encoding", "linear16"), ("sample_rate", sample_rate)]
        if eager_eot_threshold is not None:
            params.append(("eager_eot_threshold", eager_eot_threshold))  # enables EagerEndOfTurn events
        params += [("keyterm", k) for k in keyterms or []]
        self._url = f"{URL}?{urlencode(params)}"
        self._api_key = api_key
        self._chunk_bytes = int(sample_rate * CHUNK_SECONDS) * 2
        self._buffer = bytearray()
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._ws = None
        self.paused = False  # e.g. during hold music, to stop paying for audio
        self.on_event: Callable[[TurnEvent], None] = lambda event: None

    async def connect(self) -> None:
        self._ws = await websockets.connect(self._url, additional_headers={"Authorization": f"Token {self._api_key}"})

    def feed(self, pcm: bytes) -> None:
        """Queue call audio; safe to call from the audio callback."""
        if self.paused:
            return
        self._buffer += pcm
        if len(self._buffer) >= self._chunk_bytes:
            self._queue.put_nowait(bytes(self._buffer))
            self._buffer.clear()

    async def run(self) -> None:
        """Send audio and dispatch turn events until close() is called or the socket drops."""
        sender = asyncio.create_task(self._send())
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("type") == "TurnInfo":
                    self.on_event(TurnEvent(msg["event"], msg.get("transcript", ""), msg.get("turn_index", 0)))
                elif msg.get("type") == "Error":
                    log.error("Flux error %s: %s", msg.get("code"), msg.get("description"))
        except websockets.ConnectionClosed as e:
            log.info("Flux connection closed: %s", e)
        finally:
            sender.cancel()

    async def close(self) -> None:
        self._queue.put_nowait(None)
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except websockets.ConnectionClosed:
                pass
            await self._ws.close()

    async def _send(self) -> None:
        while (chunk := await self._queue.get()) is not None:
            await self._ws.send(chunk)
