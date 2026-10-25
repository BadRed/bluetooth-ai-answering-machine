"""A stand-in for phone.Phone that needs no Bluetooth: one simulated incoming call whose
caller audio comes from scripted clips or a microphone, with the assistant's side going to a WAV
file (and optionally the speakers). Same interface as phone.Phone / phone.ScoLink.
"""

import array
import asyncio
import logging
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger("answering_machine.fakephone")

SAMPLE_RATE = 8000
FRAME_SECONDS = 0.02
FRAME_BYTES = int(SAMPLE_RATE * FRAME_SECONDS) * 2


def _rms(pcm: bytes) -> float:
    samples = array.array("h", pcm)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5 if samples else 0.0


class FakeLink:
    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self, source: "Source", sink: "Sink"):
        self.on_audio: Callable[[bytes], None] = lambda pcm: None
        self.closed = asyncio.Event()
        self.assistant_quiet_since = time.monotonic()
        self._source = source
        self._sink = sink
        self._out = bytearray()
        self._task = asyncio.create_task(self._tick())

    def play(self, pcm: bytes) -> None:
        self._out += pcm

    def clear_output(self) -> None:
        self._out.clear()

    @property
    def queued(self) -> int:
        return len(self._out)

    def close(self) -> None:
        if not self.closed.is_set():
            self._task.cancel()
            self._sink.close()
            self.closed.set()

    async def _tick(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while True:
            rx = self._source.read(FRAME_BYTES, self)
            tx = bytes(self._out[:FRAME_BYTES]).ljust(FRAME_BYTES, b"\0")
            del self._out[:FRAME_BYTES]
            if _rms(tx) > 200:
                self.assistant_quiet_since = time.monotonic()
                self._source.assistant_spoke(self.assistant_quiet_since)
            self.on_audio(rx)
            self._sink.write(rx, tx)
            deadline += FRAME_SECONDS
            await asyncio.sleep(max(0.0, deadline - loop.time()))


class Source:
    finished = False

    def read(self, n: int, link: FakeLink) -> bytes:
        return b"\0" * n

    def assistant_spoke(self, now: float) -> None: ...


class ScriptSource(Source):
    """Plays caller clips in order. After each clip it waits for the assistant to reply and then go
    quiet for `gap` seconds (or gives up after `reply_timeout`), like a polite caller.

    It also logs the gap the caller actually experiences: end of their clip → the assistant's voice.
    """

    def __init__(self, clips: list[Path], gap: float = 1.2, reply_timeout: float = 12.0):
        self._clips = [self._load(p) for p in clips]
        self._gap = gap
        self._reply_timeout = reply_timeout
        self._current = b""
        self._clip_ended_at: float | None = None
        self._awaiting_reply = False
        self.response_gaps: list[float] = []

    def assistant_spoke(self, now: float) -> None:
        if self._awaiting_reply:
            self._awaiting_reply = False
            gap = now - self._clip_ended_at
            self.response_gaps.append(gap)
            log.info("Caller-perceived response time: %.0f ms (end of caller speech -> assistant audio)", gap * 1000)

    @staticmethod
    def _load(path: Path) -> bytes:
        with wave.open(str(path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError(f"{path} must be 8 kHz mono 16-bit")
            return w.readframes(w.getnframes())

    def read(self, n: int, link: FakeLink) -> bytes:
        now = time.monotonic()
        if self._awaiting_reply and now - self._clip_ended_at > self._reply_timeout:
            log.warning("No reply within %.0fs; caller carries on", self._reply_timeout)
            self._awaiting_reply = False
        quiet = now - link.assistant_quiet_since
        if not self._current and not self._awaiting_reply and self._clips and quiet > self._gap and link.queued == 0:
            self._current = self._clips.pop(0)
            log.info("Caller speaks (clip %d left)", len(self._clips))
        if not self._current:
            self.finished = not self._clips and not self._awaiting_reply
            return b"\0" * n
        chunk, self._current = self._current[:n], self._current[n:]
        if not self._current:
            self._clip_ended_at, self._awaiting_reply = now, True
        return chunk.ljust(n, b"\0")


class MicSource(Source):
    """Live microphone (needs the `mic` extra: sounddevice)."""

    def __init__(self):
        import sounddevice

        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stream = sounddevice.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                                  blocksize=160, callback=self._callback)
        self._stream.start()

    def _callback(self, data, frames, time_info, status) -> None:
        with self._lock:
            self._buf += bytes(data)

    def read(self, n: int, link: FakeLink) -> bytes:
        with self._lock:
            chunk = bytes(self._buf[:n])
            del self._buf[:n]
        return chunk.ljust(n, b"\0")


class Sink:
    def write(self, rx: bytes, tx: bytes) -> None: ...
    def close(self) -> None: ...


class StereoWavSink(Sink):
    """Left channel: caller. Right channel: the assistant."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(2)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)

    def write(self, rx: bytes, tx: bytes) -> None:
        left, right = array.array("h", rx), array.array("h", tx)
        both = array.array("h", bytes(len(rx) * 2))
        both[0::2], both[1::2] = left, right
        self._wav.writeframes(both.tobytes())

    def close(self) -> None:
        self._wav.close()


class SpeakerSink(Sink):
    """Plays the assistant's side on the speakers (needs the `mic` extra)."""

    def __init__(self):
        import sounddevice

        self._stream = sounddevice.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        self._stream.start()

    def write(self, rx: bytes, tx: bytes) -> None:
        self._stream.write(tx)

    def close(self) -> None:
        self._stream.stop()


class TeeSink(Sink):
    def __init__(self, *sinks: Sink):
        self._sinks = sinks

    def write(self, rx: bytes, tx: bytes) -> None:
        for s in self._sinks:
            s.write(rx, tx)

    def close(self) -> None:
        for s in self._sinks:
            s.close()


class FakeCall:
    def __init__(self, phone: "FakePhone", number: str):
        self.path = "/fake/voicecall01"
        self.number = number
        self.name = ""
        self.state = "incoming"
        self._phone = phone

    async def answer(self) -> None:
        self.state = "active"
        self._phone.on_call_state(self)
        await self._phone.connect_audio()

    async def hangup(self) -> None:
        self._phone._end(self)

    def __repr__(self) -> str:
        return f"<FakeCall {self.number} {self.state}>"


class FakePhone:
    def __init__(self, source: Source, sink: Sink, number: str, ring_after: float = 1.0, hangup_when_done: float = 8.0):
        self.calls: dict[str, FakeCall] = {}
        self.link: FakeLink | None = None
        self.ended = asyncio.Event()
        self.claim_audio: Callable[[], bool] = lambda: False
        self.on_call_added: Callable[[FakeCall], None] = lambda call: None
        self.on_call_state: Callable[[FakeCall], None] = lambda call: None
        self.on_call_removed: Callable[[FakeCall], None] = lambda call: None
        self.on_audio_open: Callable[[FakeLink], None] = lambda link: None
        self._source = source
        self._sink = sink
        self._number = number
        self._ring_after = ring_after
        self._hangup_when_done = hangup_when_done
        self._tasks: set[asyncio.Task] = set()

    connected = True

    async def start(self) -> None:
        self._spawn(self._ring())

    async def hangup_all(self) -> None:
        for call in list(self.calls.values()):
            self._end(call)

    async def connect_audio(self) -> None:
        if not self.claim_audio():
            log.info("Refusing audio so it stays on the (fake) handset")
            return
        self.link = FakeLink(self._source, self._sink)
        self.on_audio_open(self.link)

    def release_audio(self) -> None:
        if self.link:
            self.link.close()
            log.info("Audio released: the call would now continue on the handset")
            self._spawn(self._user_finishes_call())

    async def _user_finishes_call(self) -> None:
        await asyncio.sleep(2)
        for call in list(self.calls.values()):
            log.info("(You talk on the handset, then hang up)")
            self._end(call)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _ring(self) -> None:
        await asyncio.sleep(self._ring_after)
        call = FakeCall(self, self._number)
        self.calls[call.path] = call
        log.info("Incoming call from %s", self._number)
        self.on_call_added(call)
        self._spawn(self._caller_hangs_up_when_done(call))

    async def _caller_hangs_up_when_done(self, call: FakeCall) -> None:
        """A scripted caller hangs up once their lines are used up and the assistant has gone quiet."""
        while call.path in self.calls:
            await asyncio.sleep(0.5)
            link = self.link
            if link and self._source.finished and time.monotonic() - link.assistant_quiet_since > self._hangup_when_done:
                log.info("Caller hangs up")
                self._end(call)

    def _end(self, call: FakeCall) -> None:
        if self.calls.pop(call.path, None) is None:
            return
        if self.link:
            self.link.close()
        call.state = "disconnected"
        self.on_call_removed(call)
        self.ended.set()
