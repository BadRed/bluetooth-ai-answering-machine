"""One answered call: caller audio → Flux → (Claude | echo) → Cartesia → caller.

Handles barge-in (the caller talking over the assistant stops playback and trims Claude's history
to what was actually heard), escalation holds, and per-turn latency logging.

Speculative replies: on Flux's EagerEndOfTurn we start Claude and TTS straight away but hold
the audio back. EndOfTurn with the same transcript releases it; TurnResumed (the caller kept
talking) throws it away. This overlaps Claude's thinking time with Flux's end-of-turn wait.
"""

import array
import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import anthropic

from answering_machine.agent import ScreeningAgent, ToolCall
from answering_machine.stt import FluxSTT, TurnEvent
from answering_machine.tts import BYTES_PER_SECOND, CartesiaTTS, Utterance

log = logging.getLogger("answering_machine.session")

HOLD_REMINDER_SECONDS = 20
HOLD_TONE_EVERY = 4.0


def _hold_tone() -> bytes:
    """A soft two-note chime (8 kHz s16le) so the caller knows they're still connected."""
    samples = array.array("h")
    for freq in (660, 523):
        n = int(8000 * 0.18)
        for i in range(n):
            fade = min(1.0, i / 200, (n - i) / 200)
            samples.append(int(2500 * fade * math.sin(2 * math.pi * freq * i / 8000)))
    return samples.tobytes()


HOLD_TONE = _hold_tone()
GOODBYES = {"take_message": "Thanks, I'll pass that on. Goodbye.", "hang_up": "Goodbye."}


def _words(text: str) -> list[str]:
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).split()


def _same(a: str, b: str) -> bool:
    return _words(a) == _words(b)


BACKCHANNEL = {"yeah", "yes", "yep", "yup", "mhm", "mhmm", "mm", "mmhmm", "hmm", "uh", "um", "huh", "uhhuh",
               "ok", "okay", "right", "sure", "cool", "alright", "oh", "ah", "aha", "great", "fine", "thanks"}
BARGE_IN_WORDS = 3  # interim words of real speech before we stop talking


def _is_backchannel(text: str) -> bool:
    """'Yeah', 'mhmm', 'okay right': listening noises, not an attempt to take the turn."""
    words = _words(text)
    return 0 < len(words) <= 3 and all(w in BACKCHANNEL for w in words)


@dataclass
class Outcome:
    kind: str  # message | hung_up | handed_off | caller_hung_up | timeout
    details: dict = field(default_factory=dict)


# Returns "take", "message", "hang_up" or None (no answer in time).
Escalate = Callable[[str], Awaitable[str | None]]


class CallSession:
    def __init__(self, link, stt: FluxSTT, tts: CartesiaTTS, agent: ScreeningAgent | None, *,
                 greeting: str, escalate: Escalate, owner: str, max_seconds: float):
        self.link = link
        self.stt = stt
        self.tts = tts
        self.agent = agent  # None = echo bot
        self.greeting = greeting
        self.escalate = escalate
        self.owner = owner
        self.max_seconds = max_seconds
        self.transcript: list[dict] = []
        self.latencies: list[dict] = []

        self._started = time.monotonic()
        self._outcome: asyncio.Future[Outcome] | None = None
        self._reply_task: asyncio.Task | None = None
        self._speaking: Utterance | None = None
        self._holding = False
        self._turn_ended_at: float | None = None
        self._spec_text: str | None = None  # transcript a held (speculative) reply answers
        self._spec_utt: Utterance | None = None
        self._release_held: Callable[[], None] | None = None
        self._caller_spoke = asyncio.Event()
        self.silence_prompt: str | None = None  # said if the caller is silent after the greeting

    async def run(self) -> Outcome:
        loop = asyncio.get_running_loop()
        self._outcome = loop.create_future()
        self.link.on_audio = self.stt.feed
        self.stt.on_event = self._on_turn
        await asyncio.gather(self.stt.connect(), self.tts.connect())
        background = [asyncio.create_task(self.stt.run()), asyncio.create_task(self.tts.run()),
                      asyncio.create_task(self._watch_link())]
        try:
            await self._say(self.greeting)
            if self.silence_prompt:
                try:
                    await asyncio.wait_for(self._caller_spoke.wait(), 5.0)
                except TimeoutError:
                    log.info("Caller silent after hello; prompting")
                    await self._say(self.silence_prompt)
            return await asyncio.wait_for(asyncio.shield(self._outcome), self.max_seconds)
        except TimeoutError:
            log.warning("Call hit the %ss limit", self.max_seconds)
            return Outcome("timeout")
        finally:
            if self._reply_task:
                self._reply_task.cancel()
            for task in background:
                task.cancel()
            await asyncio.gather(self.stt.close(), self.tts.close(), return_exceptions=True)

    # Turn handling

    def _on_turn(self, event: TurnEvent) -> None:
        kind, text = event.event, event.transcript.strip()
        self._caller_spoke.set()
        talking_over = self._speaking is not None and not self._holding
        if kind == "Update":
            # Barge-in waits for a few words of real speech, so "yeah" or "mhmm" while
            # the assistant talks doesn't cut it off.
            if talking_over and len(_words(text)) >= BARGE_IN_WORDS and not _is_backchannel(text):
                self._barge_in()
        elif kind == "EagerEndOfTurn" and text and not self._holding:
            if talking_over:
                if _is_backchannel(text):
                    return
                self._barge_in()
            self._cancel_speculation()
            self._cancel_reply()
            self._spec_text = text
            # Create the held utterance now, not inside the task: EndOfTurn often arrives in
            # the same tick, and must release this reply, not an older one.
            gate = asyncio.Event()
            utt = self._utterance(gate)
            self._reply_task = asyncio.create_task(self._reply(text=text, gate=gate, utt=utt))
        elif kind == "TurnResumed":
            self._cancel_speculation()
        elif kind == "EndOfTurn" and text:
            self._log("caller", text)
            if self._holding:
                return  # on hold: logged, but nobody answers until the user decides
            if self._speaking is not None:
                if _is_backchannel(text):
                    return  # a listening noise while the assistant talks: carry on
                self._barge_in()  # short but real ("No.", "Stop.")
            self._turn_ended_at = asyncio.get_running_loop().time()
            if self._spec_text is not None and _same(self._spec_text, text):
                log.info("Eager reply confirmed")
                self._spec_text, self._spec_utt = None, None
                self._release_held()
                return
            self._cancel_speculation()
            self._cancel_reply()
            self._reply_task = asyncio.create_task(self._reply(text=text))

    def _cancel_reply(self) -> None:
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()

    def _cancel_speculation(self) -> None:
        """Throw away a held reply the caller never heard."""
        if self._spec_text is None:
            return
        self._cancel_reply()
        if self._spec_utt:
            asyncio.create_task(self._spec_utt.cancel())
        if self.agent:
            self.agent.discard_pending()
        self._spec_text, self._spec_utt = None, None

    async def _reply(self, text: str | None = None, tool_result: tuple[str, str] | None = None,
                     gate: asyncio.Event | None = None, utt: Utterance | None = None) -> list[ToolCall]:
        utt = utt or self._utterance(gate)
        if self.agent is None:
            await utt.send(text)
            await utt.finish()
            calls = []
        else:
            try:
                calls = await self.agent.respond(utt, text=text, tool_result=tool_result)
            except anthropic.APIError as e:
                log.error("Claude request failed: %s", e)
                await utt.cancel()
                if gate:
                    await gate.wait()
                await self._say("Sorry, I didn't catch that. Could you say it again?")
                return []
        if gate:
            await gate.wait()  # held until EndOfTurn confirms the caller has finished
        await self._play_out(utt)
        if utt.text:
            self._log("assistant", utt.text)
        for call in calls:
            if not utt.text and call.name in GOODBYES:  # never end a call in silence
                await self._say(GOODBYES[call.name])
            await self._tool(call)
        return calls

    async def _say(self, line: str) -> None:
        utt = self._utterance()
        await utt.send(line)
        await utt.finish()
        await self._play_out(utt)
        self._log("assistant", line)

    def _utterance(self, gate: asyncio.Event | None = None) -> Utterance:
        """A new TTS utterance wired to the call. With a gate, audio is held until it opens."""
        utt = self.tts.utterance()
        held: list[bytes] = []
        started = False

        def play(pcm: bytes) -> None:
            nonlocal started
            if not started:
                started = True
                self._log_latency(utt)
            self.link.play(pcm)

        def on_audio(pcm: bytes) -> None:
            if gate is not None and not gate.is_set():
                held.append(pcm)
            else:
                play(pcm)

        utt.on_audio = on_audio
        if gate is None:
            self._speaking = utt
        else:
            self._spec_utt = utt

            def release() -> None:
                self._speaking = utt
                gate.set()
                for pcm in held:
                    play(pcm)
                held.clear()

            self._release_held = release
        return utt

    def _log_latency(self, utt: Utterance) -> None:
        turn_ended_at, self._turn_ended_at = self._turn_ended_at, None
        if turn_ended_at is None:
            return  # not a reply to a caller turn (greeting, hold line, tool follow-up)
        now = asyncio.get_running_loop().time()
        first_text = utt.first_text_at - turn_ended_at if utt.first_text_at else None
        entry = {"turn": len(self.latencies) + 1, "eot_to_first_text": first_text,
                 "eot_to_first_audio": now - turn_ended_at}
        self.latencies.append(entry)
        log.info("Latency turn %d: first text %s, first audio %.0f ms", entry["turn"],
                 f"{first_text * 1000:.0f} ms" if first_text is not None else "n/a",
                 entry["eot_to_first_audio"] * 1000)

    async def _play_out(self, utt: Utterance) -> None:
        """Wait until all of the utterance's audio has been generated and sent to the caller."""
        await utt.done.wait()
        while self.link.queued and not self.link.closed.is_set():
            await asyncio.sleep(0.02)
        if self._speaking is utt:
            self._speaking = None

    def _barge_in(self) -> None:
        utt = self._speaking
        if utt.done.is_set() and not self.link.queued:
            self._speaking = None  # it had already finished playing: nothing to interrupt
            return
        played = max(0.0, (utt.audio_bytes - self.link.queued) / BYTES_PER_SECOND)
        self.link.clear_output()
        self._speaking = None
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()
        asyncio.create_task(utt.cancel())
        spoken = utt.spoken_text(played)
        log.info("Barge-in after %.1fs; caller heard: %r", played, spoken)
        if self.agent:
            self.agent.interrupted(spoken)
        if spoken:
            self._log("assistant", spoken + " —")

    # Tools

    async def _tool(self, call: ToolCall) -> None:
        log.info("Tool %s %s", call.name, call.input)
        if call.name == "take_message":
            self._finish(Outcome("message", call.input))
        elif call.name == "hang_up":
            self._finish(Outcome("hung_up", call.input))
        elif call.name == "escalate":
            await self._escalate(call)

    async def _escalate(self, call: ToolCall) -> None:
        self._holding = True
        decision_task = asyncio.create_task(self.escalate(call.input["announcement"]))
        tone_task = asyncio.create_task(self._hold_tone())
        try:
            while True:
                done, _ = await asyncio.wait([decision_task], timeout=HOLD_REMINDER_SECONDS)
                if done:
                    break
                await self._say("Thanks for holding, I'm still trying to reach them.")
            decision = decision_task.result()
        finally:
            decision_task.cancel()
            tone_task.cancel()
            self._holding = False

        log.info("Escalation decision: %s", decision)
        if decision == "take":
            self._finish(Outcome("handed_off", call.input))
            return
        if decision == "hang_up":
            result = "The person they're calling asked you to end the call. Say a brief, polite goodbye, then call hang_up."
        else:
            result = "The person they're calling can't take the call right now. Let the caller know and take a message."
        await self._reply(tool_result=(call.id, result))

    async def _hold_tone(self) -> None:
        while True:
            await asyncio.sleep(HOLD_TONE_EVERY)
            if not self.link.queued:  # don't step on a hold reminder
                self.link.play(HOLD_TONE)

    # Bookkeeping

    async def _watch_link(self) -> None:
        await self.link.closed.wait()
        self._finish(Outcome("caller_hung_up"))

    def _finish(self, outcome: Outcome) -> None:
        if self._outcome and not self._outcome.done():
            self._outcome.set_result(outcome)

    def _log(self, who: str, text: str) -> None:
        self.transcript.append({"t": round(time.monotonic() - self._started, 1), "who": who, "text": text})

    def transcript_text(self) -> str:
        return "\n".join(f"[{e['t']:>5}s] {e['who']}: {e['text']}" for e in self.transcript)
