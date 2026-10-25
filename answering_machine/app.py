"""The answering machine: decides which calls to answer, runs a session for each, and reports to the user."""

import asyncio
import json
import logging
import time
from pathlib import Path

import anthropic

from answering_machine import contacts
from answering_machine.agent import ScreeningAgent, summarise
from answering_machine.config import Config
from answering_machine.dgagent import DeepgramAgentSession
from answering_machine.numbers import NumberSet, normalise
from answering_machine.prompts import greeting, silence_prompt
from answering_machine.session import CallSession, Outcome
from answering_machine.stt import FluxSTT
from answering_machine.tts import CartesiaTTS

log = logging.getLogger("answering_machine.app")

OUTCOME_LABELS = {
    "message": "message taken",
    "hung_up": "hung up (spam/scam)",
    "handed_off": "handed to you",
    "caller_hung_up": "caller hung up",
    "timeout": "hit time limit",
    "error": "voice agent error",
}


class AnsweringMachine:
    def __init__(self, cfg: Config, phone, notifier, *, echo: bool = False):
        self.cfg = cfg
        self.phone = phone
        self.notifier = notifier
        self.echo = echo
        self.paused = False
        self.client = anthropic.AsyncAnthropic()
        self.calls_done = 0

        self._owned: str | None = None  # path of the call we answered
        self._handed_off = False
        self._session: CallSession | None = None
        self._tasks: set[asyncio.Task] = set()
        self._allow = NumberSet({n: "allowlist" for n in cfg.allowlist})
        self._contacts = NumberSet()
        self._contacts_mtime = -1.0

        phone.claim_audio = lambda: self._owned is not None and not self._handed_off
        phone.on_call_added = self._call_added
        phone.on_call_state = lambda call: log.info("State: %r", call)
        phone.on_call_removed = self._call_removed
        phone.on_audio_open = self._audio_open
        notifier.on_command = self.command

    async def run(self) -> None:
        await self.phone.start()
        await self.notifier.run()

    async def wait_idle(self) -> None:
        while self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _known(self, number: str) -> str | None:
        """'allowlist' or the contact's name if the number is known (matched on its last digits), else None."""
        path = contacts.contacts_path(self.cfg.data_dir)
        mtime = path.stat().st_mtime if path.exists() else 0.0
        if mtime != self._contacts_mtime:  # re-read after a sync, no restart needed
            self._contacts, self._contacts_mtime = contacts.load(self.cfg.data_dir), mtime
            log.info("Loaded %d known numbers from contacts", len(self._contacts))
        found = self._allow.lookup(number)
        return found if found is not None else self._contacts.lookup(number)

    # Phone events

    def _call_added(self, call) -> None:
        number = normalise(call.number)
        log.info("Call added: %r", call)
        if call.state != "incoming":
            return  # outgoing call from the handset: not ours
        if self.cfg.screen_all:
            log.info("Screening all calls (testing): answering %s%s", number, f" ({call.name})" if call.name else "")
            self._spawn(self._answer_after_delay(call))
        elif call.name:
            log.info("%s is a contact (%s); letting it ring", number, call.name)
        elif (known := self._known(call.number)) is not None:
            log.info("%s is known (%s); letting it ring", number, known)
        elif self.paused:
            log.info("Screening paused; letting it ring")
        elif self._session:
            log.info("Already on a call; ignoring call waiting")
        else:
            self._spawn(self._answer_after_delay(call))

    async def _answer_after_delay(self, call) -> None:
        await asyncio.sleep(self.cfg.ring_delay)
        if call.path not in self.phone.calls or call.state != "incoming":
            log.info("Not answering %r: picked up on the handset or gone", call)
            return
        if call.name and not self.cfg.screen_all:  # the contact name can arrive after the call itself
            log.info("%r is a contact; letting it ring", call)
            return
        self._owned, self._handed_off = call.path, False
        log.info("Answering %r", call)
        await call.answer()
        await asyncio.sleep(2)
        link = self.phone.link
        if self._owned == call.path and (link is None or link.closed.is_set()):
            log.info("Phone didn't open audio; requesting it")
            await self.phone.connect_audio()

    def _audio_open(self, link) -> None:
        if self._session:
            return
        self._spawn(self._screen(link, self.phone.calls.get(self._owned)))

    def _call_removed(self, call) -> None:
        log.info("Call removed: %r", call)
        if call.path == self._owned:
            self._owned = None

    # A screened call

    async def _screen(self, link, call) -> None:
        number = call.number if call else ""
        cfg = self.cfg
        escalate = lambda announcement: self._ask_user(number, announcement)
        if cfg.engine == "deepgram" and not self.echo:
            if cfg.dg_speak_provider == "cartesia":
                speak = DeepgramAgentSession.speak_cartesia(cfg.cartesia_api_key, cfg.cartesia_voice_id, cfg.cartesia_model)
            else:
                speak = DeepgramAgentSession.speak_deepgram(cfg.dg_voice)
            session = DeepgramAgentSession(
                link, api_key=cfg.deepgram_api_key, think_provider=cfg.dg_think_provider,
                think_model=cfg.dg_think_model, speak=speak, owner=cfg.owner_name, caller_number=number,
                greeting=greeting(cfg.owner_name), escalate=escalate, max_seconds=cfg.max_call_seconds)
        else:
            session = self._local_session(link, number, escalate)
        self._session = session
        started = time.time()
        try:
            outcome = await session.run()
        finally:
            self._session = None
        log.info("Call outcome: %s %s", outcome.kind, outcome.details)

        if outcome.kind == "handed_off":
            self._handed_off = True  # first, so the phone can't hand audio straight back
            self.phone.release_audio()
        elif outcome.kind != "caller_hung_up" and call and call.path in self.phone.calls:
            await call.hangup()
        await self._report(number, session, outcome, time.time() - started)

    def _local_session(self, link, number: str, escalate) -> CallSession:
        """Our own pipeline: Flux + Claude + Cartesia, wired together in session.py."""
        cfg = self.cfg
        agent = None if self.echo else ScreeningAgent(self.client, cfg.anthropic_model, cfg.owner_name, number)
        session = CallSession(
            link,
            FluxSTT(cfg.deepgram_api_key, keyterms=[cfg.owner_name]),
            CartesiaTTS(cfg.cartesia_api_key, cfg.cartesia_voice_id, cfg.cartesia_model),
            agent,
            greeting="Echo test. Say something and I'll repeat it." if self.echo else greeting(cfg.owner_name),
            escalate=escalate,
            owner=cfg.owner_name,
            max_seconds=cfg.max_call_seconds,
        )
        if not self.echo:
            session.silence_prompt = silence_prompt(cfg.owner_name)
        return session

    async def _ask_user(self, number: str, announcement: str) -> str | None:
        return await self.notifier.ask(
            f"📞 {announcement}\nFrom {number or 'a withheld number'}. They're on hold.",
            [("Take it", "take"), ("Take a message", "message"), ("Hang up", "hang_up")],
            self.cfg.escalation_timeout,
        )

    async def _report(self, number: str, session: CallSession, outcome: Outcome, duration: float) -> None:
        self.calls_done += 1
        heard_caller = any(e["who"] == "caller" for e in session.transcript)
        summary = ""
        if not self.echo and heard_caller:
            summary = await summarise(self.client, self.cfg.summary_model, session.transcript_text(),
                                      OUTCOME_LABELS.get(outcome.kind, outcome.kind))

        record = {
            "number": number, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - duration)),
            "duration": round(duration), "outcome": outcome.kind, "details": outcome.details,
            "summary": summary, "transcript": session.transcript, "latencies": session.latencies,
        }
        out = Path(self.cfg.data_dir) / "calls" / time.strftime("%Y%m%d-%H%M%S.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2))
        log.info("Saved %s", out)

        lines = [f"📞 {number or 'Withheld'} · {OUTCOME_LABELS.get(outcome.kind, outcome.kind)} · {duration:.0f}s"]
        if summary:
            lines.append(summary)
        if outcome.kind == "message":
            d = dict(outcome.details)
            digits = "".join(c for c in d.get("callback_number", "") if c.isdigit())
            if digits and len(digits) < 10:
                d["callback_number"] += " (looks incomplete, check the transcript)"
            lines.append("\n".join(f"{k.replace('_', ' ').capitalize()}: {d[k]}" for k in
                                   ("caller_name", "organisation", "reason", "callback_number", "urgency") if d.get(k)))
        if session.latencies:
            avg = sum(t["eot_to_first_audio"] for t in session.latencies) / len(session.latencies)
            lines.append(f"Avg response latency: {avg * 1000:.0f} ms over {len(session.latencies)} turns")
        try:
            await self.notifier.notify("\n\n".join(lines))
            await self.notifier.notify("Transcript:\n" + session.transcript_text()[-3500:])
        except Exception as e:  # a failed notification mustn't lose the call record
            log.error("Couldn't send report: %s", e)

    # User commands (Telegram)

    async def command(self, cmd: str, args: str) -> str:
        if cmd == "pause":
            self.paused = True
            return "Screening paused. Calls will ring normally."
        if cmd == "resume":
            self.paused = False
            return "Screening resumed."
        if cmd == "status":
            state = "on a call" if self._session else "idle"
            return (f"Phone: {'connected' if self.phone.connected else 'not connected'}\n"
                    f"Screening: {'paused' if self.paused else 'on'} ({state})\n"
                    f"Known numbers: {len(self._allow)} allowlisted, {len(self._contacts)} from contacts · "
                    f"ring delay {self.cfg.ring_delay:.0f}s")
        return "Commands: /status, /pause, /resume"

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)

        def done(t: asyncio.Task) -> None:
            self._tasks.discard(t)
            if not t.cancelled() and t.exception():
                log.error("Task failed", exc_info=t.exception())

        task.add_done_callback(done)
