"""Phase 1 spike: log calls, answer/hang up/dial, record and play call audio, and test the
audio ownership rule.

Run on the host as root, e.g.:
  uv run spike.py --answer-after 6 --play assets/test-clip.wav --hangup-after 25
  uv run spike.py --answer-after 6 --release-after 10      # handoff test
  uv run spike.py                                          # just log; the assistant never takes audio

Interactive commands (when run in a terminal): a=answer, h=hang up all, d <number>=dial,
t <digits>=send tones, r=release audio, c=ask phone to open audio, q=quit.
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
import wave
from pathlib import Path

from answering_machine.phone import Call, Phone, ScoLink

log = logging.getLogger("answering_machine.spike")


class Spike:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.phone = Phone()
        self.owned: str | None = None  # path of the call the assistant answered or dialled
        self.released = False
        self.clip = self._load_clip(args.play) if args.play else b""
        self.recording: wave.Wave_write | None = None

        self.phone.claim_audio = lambda: self.owned is not None and not self.released
        self.phone.on_call_added = self.call_added
        self.phone.on_call_state = lambda call: log.info("State: %r", call)
        self.phone.on_call_removed = self.call_removed
        self.phone.on_audio_open = self.audio_open

    @staticmethod
    def _load_clip(path: str) -> bytes:
        with wave.open(path, "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (ScoLink.SAMPLE_RATE, 1, 2):
                sys.exit(f"{path} must be 8 kHz mono 16-bit")
            return w.readframes(w.getnframes())

    def call_added(self, call: Call) -> None:
        log.info("Call added: %r", call)
        if call.state == "incoming" and self.args.answer_after is not None:
            asyncio.create_task(self.answer_later(call))

    async def answer_later(self, call: Call) -> None:
        await asyncio.sleep(self.args.answer_after)
        if call.path not in self.phone.calls or call.state != "incoming":
            log.info("Not answering %r (picked up elsewhere or gone)", call)
            return
        await self.answer(call)

    async def answer(self, call: Call) -> None:
        self.owned, self.released = call.path, False
        log.info("Answering %r", call)
        await call.answer()
        await asyncio.sleep(2)
        if not self.phone.link or self.phone.link.closed.is_set():
            log.info("Phone didn't open audio; requesting it")
            await self.phone.connect_audio()

    async def dial(self, number: str) -> None:
        self.released = False
        self.owned = await self.phone.dial(number)
        log.info("Dialled %s -> %s", number, self.owned)

    def audio_open(self, link: ScoLink) -> None:
        out_dir = Path(self.args.record_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / time.strftime("%Y%m%d-%H%M%S-rx.wav")
        self.recording = wave.open(str(path), "wb")
        self.recording.setnchannels(1)
        self.recording.setsampwidth(2)
        self.recording.setframerate(ScoLink.SAMPLE_RATE)
        link.on_audio = self.recording.writeframes
        log.info("Recording caller audio to %s", path)

        if self.clip:
            link.play(self.clip)
        if self.args.release_after is not None:
            asyncio.create_task(self.release_later(link))
        if self.args.hangup_after is not None:
            asyncio.create_task(self.hangup_later(link))
        asyncio.create_task(self.stop_recording(link))

    async def release_later(self, link: ScoLink) -> None:
        await asyncio.sleep(self.args.release_after)
        log.info("Releasing audio to the handset")
        self.release()

    async def hangup_later(self, link: ScoLink) -> None:
        await asyncio.sleep(self.args.hangup_after)
        if not link.closed.is_set():
            log.info("Hanging up")
            await self.phone.hangup_all()

    async def stop_recording(self, link: ScoLink) -> None:
        await link.closed.wait()
        if self.recording:
            self.recording.close()
            self.recording = None

    def release(self) -> None:
        self.released = True
        self.phone.release_audio()

    def call_removed(self, call: Call) -> None:
        log.info("Call removed: %r", call)
        if call.path == self.owned:
            self.owned = None

    async def command(self, line: str) -> None:
        cmd, _, arg = line.strip().partition(" ")
        try:
            if cmd == "a":
                incoming = [c for c in self.phone.calls.values() if c.state == "incoming"]
                if incoming:
                    await self.answer(incoming[0])
            elif cmd == "h":
                await self.phone.hangup_all()
            elif cmd == "d" and arg:
                await self.dial(arg)
            elif cmd == "t" and arg:
                await self.phone.send_tones(arg)
            elif cmd == "r":
                self.release()
            elif cmd == "c":
                await self.phone.connect_audio()
            elif cmd == "q":
                raise SystemExit
            elif cmd:
                print("a | h | d <number> | t <digits> | r | c | q")
        except SystemExit:
            raise
        except Exception as e:
            log.error("%s failed: %s", cmd, e)

    async def run(self) -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        startup = asyncio.create_task(self.phone.start())
        await asyncio.wait([startup, asyncio.create_task(stop.wait())], return_when=asyncio.FIRST_COMPLETED)
        if stop.is_set():
            startup.cancel()
            return
        startup.result()
        if self.args.dial:
            await self.dial(self.args.dial)

        if sys.stdin.isatty():
            def on_stdin() -> None:
                line = sys.stdin.readline()
                task = asyncio.create_task(self.command(line))
                task.add_done_callback(lambda t: t.exception() and stop.set())
            loop.add_reader(sys.stdin, on_stdin)

        await stop.wait()
        self.phone.release_audio()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answer-after", type=float, help="auto-answer incoming calls after N seconds")
    parser.add_argument("--play", help="8 kHz mono WAV to play into the call once audio opens")
    parser.add_argument("--hangup-after", type=float, help="hang up N seconds after audio opens")
    parser.add_argument("--release-after", type=float, help="release audio to the handset after N seconds")
    parser.add_argument("--dial", help="dial this number on startup")
    parser.add_argument("--record-dir", default="data/recordings")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    asyncio.run(Spike(args).run())


if __name__ == "__main__":
    main()
