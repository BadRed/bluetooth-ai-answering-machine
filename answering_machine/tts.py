"""Cartesia streaming text-to-speech over one WebSocket per call.

Each reply is an Utterance (a Cartesia context): text is sent in pieces as Claude streams it,
audio comes back as raw 8 kHz s16le PCM, and word timestamps let us work out what the caller
actually heard if they interrupt.
"""

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import Callable

import websockets

log = logging.getLogger("answering_machine.tts")

URL = "wss://api.cartesia.ai/tts/websocket?cartesia_version=2026-08-14"
SAMPLE_RATE = 8000
BYTES_PER_SECOND = SAMPLE_RATE * 2


class Utterance:
    def __init__(self, tts: "CartesiaTTS"):
        self._tts = tts
        self.context_id = uuid.uuid4().hex
        self.on_audio: Callable[[bytes], None] = lambda pcm: None
        self.words: list[tuple[str, float]] = []  # (word, start seconds)
        self.audio_bytes = 0
        self.done = asyncio.Event()
        self.text = ""
        self.first_text_at: float | None = None
        self._sent_any = False

    async def send(self, text: str) -> None:
        if not text:
            return
        if self.first_text_at is None:
            self.first_text_at = asyncio.get_running_loop().time()
        self.text += text
        self._sent_any = True
        await self._tts._request(self, text, more=True)

    async def finish(self) -> None:
        """Flush: no more text is coming for this utterance."""
        if not self._sent_any:
            self.done.set()
            return
        await self._tts._request(self, "", more=False)

    async def cancel(self) -> None:
        if self.done.is_set() or not self._sent_any:
            self.done.set()
            return  # Cartesia has no live context for this; cancelling would be an error
        self.done.set()
        await self._tts._send({"context_id": self.context_id, "cancel": True})

    def spoken_text(self, played_seconds: float) -> str:
        """Words whose audio started before `played_seconds`."""
        return " ".join(w for w, start in self.words if start < played_seconds)

    def _audio(self, pcm: bytes) -> None:
        self.audio_bytes += len(pcm)
        self.on_audio(pcm)


class CartesiaTTS:
    def __init__(self, api_key: str, voice_id: str, model_id: str = "sonic-3.6", language: str = "en"):
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._language = language
        self._ws = None
        self._utterances: dict[str, Utterance] = {}

    async def connect(self) -> None:
        self._ws = await websockets.connect(URL, additional_headers={"X-API-Key": self._api_key})

    def utterance(self) -> Utterance:
        utt = Utterance(self)
        self._utterances[utt.context_id] = utt
        return utt

    async def run(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                utt = self._utterances.get(msg.get("context_id", ""))
                if utt is None:
                    continue
                kind = msg.get("type")
                if kind == "chunk" and not utt.done.is_set():
                    utt._audio(base64.b64decode(msg["data"]))
                elif kind == "timestamps":
                    ts = msg["word_timestamps"]
                    utt.words += zip(ts["words"], ts["start"])
                elif kind == "done":
                    utt.done.set()
                    self._utterances.pop(utt.context_id, None)
                elif kind == "error":
                    log.error("Cartesia error: %s %s", msg.get("title"), msg.get("message"))
                    utt.done.set()
        except websockets.ConnectionClosed as e:
            log.info("Cartesia connection closed: %s", e)
        finally:
            for utt in self._utterances.values():
                utt.done.set()

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()

    async def _request(self, utt: Utterance, text: str, more: bool) -> None:
        await self._send({
            "model_id": self._model_id,
            "transcript": text,
            "voice": {"id": self._voice_id},
            "language": self._language,
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
            "context_id": utt.context_id,
            "continue": more,
            "add_timestamps": True,
        })

    async def _send(self, msg: dict) -> None:
        await self._ws.send(json.dumps(msg))
