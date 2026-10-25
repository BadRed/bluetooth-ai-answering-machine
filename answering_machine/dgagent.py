"""Deepgram Voice Agent engine: one WebSocket per call does speech-to-text (Flux), the LLM
and text-to-speech, including turn-taking and barge-in. We just bridge the call audio
and handle the tools (hold, handoff, message, hang up).

Same interface as session.CallSession: run() -> Outcome, plus transcript and latencies.
"""

import asyncio
import json
import logging
import re
import time

import websockets

from answering_machine.agent import _valid
from answering_machine.prompts import TOOLS, screening_system, silence_prompt
from answering_machine.session import GOODBYES, HOLD_REMINDER_SECONDS, HOLD_TONE, HOLD_TONE_EVERY, Escalate, Outcome

log = logging.getLogger("answering_machine.dgagent")

URL = "wss://agent.deepgram.com/v1/agent/converse"
SAMPLE_RATE = 8000
SEND_BYTES = int(SAMPLE_RATE * 0.04) * 2  # 40 ms chunks
GOODBYE_RE = re.compile(r"\b(good)?bye\b|\btake care\b", re.IGNORECASE)
QUIET_TO_FINISH = 0.8
SILENCE_FOLLOW_UP = 5.0  # seconds after "Hello?" before prompting a silent caller  # seconds of no agent audio before we consider a goodbye fully spoken


class DeepgramAgentSession:
    def __init__(self, link, *, api_key: str, think_provider: str, think_model: str, speak: dict,
                 owner: str, caller_number: str, greeting: str, escalate: Escalate, max_seconds: float):
        self.link = link
        self.escalate = escalate
        self.owner = owner
        self.max_seconds = max_seconds
        self.transcript: list[dict] = []
        self.latencies: list[dict] = []

        self._api_key = api_key
        self._settings = self._build_settings(think_provider, think_model, speak, owner, caller_number, greeting)
        self._started = time.monotonic()
        self._outcome: asyncio.Future[Outcome] | None = None
        self._ws = None
        self._ready = asyncio.Event()
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._buffer = bytearray()
        self._holding = False
        self._spoke_since_caller = False
        self._last_agent_audio = 0.0
        self._message: dict | None = None
        self._assistant_since_caller: list[str] = []
        self._goodbye_said = asyncio.Event()  # set when the agent speaks after take_message
        self._tasks: set[asyncio.Task] = set()
        self._caller_spoke = asyncio.Event()

    @staticmethod
    def speak_deepgram(voice: str) -> dict:
        return {"provider": {"type": "deepgram", "model": voice}}

    @staticmethod
    def speak_cartesia(api_key: str, voice_id: str, model_id: str) -> dict:
        """Cartesia voice, called by Deepgram with our Cartesia key."""
        return {
            "provider": {"type": "cartesia", "model_id": model_id, "voice": {"mode": "id", "id": voice_id},
                         "language": "en"},
            "endpoint": {"url": "https://api.cartesia.ai/tts/bytes", "headers": {"x-api-key": api_key}},
        }

    @staticmethod
    def _build_settings(provider, model, speak, owner, number, greeting) -> dict:
        functions = [{"name": t["name"], "description": t["description"], "parameters": t["input_schema"],
                      "defer_until_eot": True} for t in TOOLS]
        return {
            "type": "Settings",
            "audio": {
                "input": {"encoding": "linear16", "sample_rate": SAMPLE_RATE},
                "output": {"encoding": "linear16", "sample_rate": SAMPLE_RATE, "container": "none"},
            },
            "agent": {
                "language": "en",
                "listen": {"provider": {"type": "deepgram", "version": "v2", "model": "flux-general-en",
                                        "keyterms": [owner]}},
                "think": {"provider": {"type": provider, "model": model},
                          "prompt": f"{screening_system(owner)}\n\nCaller ID: {number or 'withheld'}",
                          "functions": functions},
                "speak": speak,
                "greeting": greeting,
            },
        }

    async def run(self) -> Outcome:
        self._outcome = asyncio.get_running_loop().create_future()
        self._ws = await websockets.connect(URL, additional_headers={"Authorization": f"Token {self._api_key}"},
                                            max_size=None)
        await self._ws.send(json.dumps(self._settings))
        self.link.on_audio = self._feed
        background = [asyncio.create_task(c) for c in (self._receive(), self._send(), self._watch_link(),
                                                       self._prompt_if_silent())]
        try:
            return await asyncio.wait_for(asyncio.shield(self._outcome), self.max_seconds)
        except TimeoutError:
            log.warning("Call hit the %ss limit", self.max_seconds)
            return Outcome("timeout")
        finally:
            for task in background + list(self._tasks):
                task.cancel()
            await self._ws.close()

    # Audio

    def _feed(self, pcm: bytes) -> None:
        if not self._ready.is_set():
            return
        # On hold the agent mustn't hear (or answer) the caller, but the stream must keep flowing.
        self._buffer += b"\0" * len(pcm) if self._holding else pcm
        if len(self._buffer) >= SEND_BYTES:
            self._queue.put_nowait(bytes(self._buffer))
            self._buffer.clear()

    async def _send(self) -> None:
        while True:
            await self._ws.send(await self._queue.get())

    # Server messages

    async def _receive(self) -> None:
        try:
            async for msg in self._ws:
                if isinstance(msg, bytes):
                    self._last_agent_audio = time.monotonic()
                    self.link.play(msg)
                    continue
                self._handle(json.loads(msg))
        except websockets.ConnectionClosed as e:
            log.warning("Deepgram agent connection closed: %s", e)
            self._finish(Outcome("error", {"reason": str(e)}))

    def _handle(self, m: dict) -> None:
        kind = m.get("type")
        if kind == "SettingsApplied":
            self._ready.set()
        elif kind == "UserStartedSpeaking":
            self._caller_spoke.set()
            if self.link.queued and not self._holding:
                log.info("Barge-in: dropping %.1fs of queued agent audio", self.link.queued / (SAMPLE_RATE * 2))
                self.link.clear_output()
        elif kind == "ConversationText":
            who = "assistant" if m.get("role") == "assistant" else "caller"
            self._log(who, m.get("content", ""))
            self._spoke_since_caller = who == "assistant"
            if who == "caller":
                self._assistant_since_caller = []
            else:
                self._assistant_since_caller.append(m.get("content", ""))
                if self._message is not None:
                    self._goodbye_said.set()
        elif kind == "AgentStartedSpeaking":
            entry = {"turn": len(self.latencies) + 1, "eot_to_first_audio": float(m.get("total_latency") or 0),
                     "llm": float(m.get("ttt_latency") or 0), "tts": float(m.get("tts_latency") or 0)}
            self.latencies.append(entry)
            log.info("Latency turn %d: %.0f ms (LLM %.0f, TTS %.0f)", entry["turn"], entry["eot_to_first_audio"] * 1000,
                     entry["llm"] * 1000, entry["tts"] * 1000)
        elif kind == "FunctionCallRequest":
            for f in m.get("functions", []):
                if f.get("client_side", True):
                    self._spawn(self._function(f))
        elif kind == "Error":
            log.error("Deepgram agent error %s: %s", m.get("code"), m.get("description"))
        elif kind == "Warning":
            log.warning("Deepgram agent warning %s: %s", m.get("code"), m.get("description"))

    async def _prompt_if_silent(self) -> None:
        await self._ready.wait()
        try:
            await asyncio.wait_for(self._caller_spoke.wait(), SILENCE_FOLLOW_UP)
        except TimeoutError:
            log.info("Caller silent after hello; prompting")
            await self._inject(silence_prompt(self.owner))

    # Tools

    async def _function(self, f: dict) -> None:
        name = f.get("name", "")
        try:
            args = json.loads(f.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = None
        log.info("Tool %s %s", name, args)
        if not _valid(name, args):
            await self._respond(f, "Invalid arguments; try again.")
            return

        if name == "take_message":
            # Don't trust the model to have said goodbye already: ask for it, and only hang up
            # once it has been spoken and heard.
            await self._drain()  # the rest of this turn's speech can arrive after the tool call
            if any(GOODBYE_RE.search(t) for t in self._assistant_since_caller):
                await self._respond(f, "Message saved.")
            else:
                self._message = args
                self._goodbye_said.clear()
                await self._respond(f, "Message saved. Now say a short, warm goodbye to the caller. "
                                       "The call ends as soon as you've said it.")
                try:
                    await asyncio.wait_for(self._goodbye_said.wait(), 10)
                except TimeoutError:
                    await self._inject(GOODBYES[name])
                await self._drain()
            self._finish(Outcome("message", args))
        elif name == "hang_up":
            await self._drain()  # the agent's text for this turn can arrive after the tool call
            if not self._spoke_since_caller:  # never end a call in silence
                await self._inject(GOODBYES[name])
                await self._drain()
            self._finish(Outcome("hung_up", args))
        elif name == "escalate":
            await self._drain()
            decision = await self._hold(args["announcement"])
            log.info("Escalation decision: %s", decision)
            if decision == "take":
                self._finish(Outcome("handed_off", args))
                return
            if decision == "hang_up":
                result = "The person they're calling asked you to end the call. Say a brief, polite goodbye, then call hang_up."
            else:
                result = "The person they're calling can't take the call right now. Let the caller know and take a message."
            await self._respond(f, result)

    async def _hold(self, announcement: str) -> str | None:
        self._holding = True
        decision = asyncio.create_task(self.escalate(announcement))
        started = time.monotonic()
        last_reminder = started
        try:
            while not decision.done():
                await asyncio.wait([decision], timeout=HOLD_TONE_EVERY)
                if decision.done():
                    break
                if time.monotonic() - last_reminder >= HOLD_REMINDER_SECONDS:
                    last_reminder = time.monotonic()
                    await self._inject("Thanks for holding, I'm still trying to reach them.")
                elif not self.link.queued:
                    self.link.play(HOLD_TONE)
            return decision.result()
        finally:
            decision.cancel()
            self._holding = False

    async def _drain(self, timeout: float = 15.0) -> None:
        """Wait until the agent has finished speaking and the caller has heard it all."""
        deadline = time.monotonic() + timeout
        await asyncio.sleep(0.3)  # let any final audio for this turn start arriving
        while time.monotonic() < deadline:
            quiet = time.monotonic() - self._last_agent_audio
            if quiet > QUIET_TO_FINISH and not self.link.queued:
                return
            await asyncio.sleep(0.05)

    async def _respond(self, f: dict, content: str) -> None:
        await self._ws.send(json.dumps({"type": "FunctionCallResponse", "id": f.get("id"),
                                        "name": f.get("name"), "content": content}))

    async def _inject(self, text: str) -> None:
        await self._ws.send(json.dumps({"type": "InjectAgentMessage", "message": text}))
        self._log("assistant", text)

    # Bookkeeping

    async def _watch_link(self) -> None:
        await self.link.closed.wait()
        self._finish(Outcome("caller_hung_up"))

    def _finish(self, outcome: Outcome) -> None:
        if self._outcome and not self._outcome.done():
            self._outcome.set_result(outcome)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _log(self, who: str, text: str) -> None:
        if text:
            self.transcript.append({"t": round(time.monotonic() - self._started, 1), "who": who, "text": text})

    def transcript_text(self) -> str:
        return "\n".join(f"[{e['t']:>5}s] {e['who']}: {e['text']}" for e in self.transcript)
