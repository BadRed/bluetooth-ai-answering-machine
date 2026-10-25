"""CallSession logic with stubbed STT/TTS/Claude: no network, no keys."""

import asyncio

from answering_machine.agent import ToolCall
from answering_machine.fakephone import FakeLink, Sink, Source
from answering_machine.contacts import parse_vcards
from answering_machine.numbers import NumberSet, normalise
from answering_machine.session import CallSession
from answering_machine.stt import TurnEvent
from answering_machine.tts import BYTES_PER_SECOND, Utterance

WORD_SECONDS = 0.25


class StubSTT:
    on_event = None

    async def connect(self): ...
    async def run(self): await asyncio.Event().wait()
    async def close(self): ...
    def feed(self, pcm): ...

    def say(self, text: str) -> None:
        self.on_event(TurnEvent("StartOfTurn", "", 0))
        self.on_event(TurnEvent("EndOfTurn", text, 0))

    def event(self, kind: str, text: str = "") -> None:
        self.on_event(TurnEvent(kind, text, 0))


class StubTTS:
    """Turns each word into WORD_SECONDS of loud audio, with word timestamps."""

    async def connect(self): ...
    async def run(self): await asyncio.Event().wait()
    async def close(self): ...

    def utterance(self) -> Utterance:
        return Utterance(self)

    async def _request(self, utt: Utterance, text: str, more: bool) -> None:
        if more:
            return
        for i, word in enumerate(utt.text.split()):
            utt.words.append((word, i * WORD_SECONDS))
            utt._audio(b"\x10\x27" * int(BYTES_PER_SECOND * WORD_SECONDS / 2))
        utt.done.set()

    async def _send(self, msg): ...


class StubAgent:
    def __init__(self, replies):
        self.replies = list(replies)  # (text, [ToolCall]) per turn
        self.heard = []  # user texts / tool results
        self.interruptions = []

    async def respond(self, utt, text=None, tool_result=None):
        self.heard.append(text if tool_result is None else tool_result)
        reply, calls = self.replies.pop(0)
        await utt.send(reply)
        await utt.finish()
        return calls

    def interrupted(self, spoken):
        self.interruptions.append(spoken)

    def discard_pending(self):
        self.discarded = getattr(self, "discarded", 0) + 1


def make_session(agent, decision=None):
    link = FakeLink(Source(), Sink())
    stt = StubSTT()

    async def escalate(announcement):
        return decision

    session = CallSession(link, stt, StubTTS(), agent, greeting="Hello there.", escalate=escalate,
                          owner="BadRed", max_seconds=20)
    return session, stt, link


async def after(seconds, fn):
    await asyncio.sleep(seconds)
    fn()


def test_take_message():
    async def go():
        agent = StubAgent([("Thanks Sarah, I'll pass that on. Bye.",
                            [ToolCall("t1", "take_message", {"caller_name": "Sarah", "reason": "blood test"})])])
        session, stt, _ = make_session(agent)
        asyncio.create_task(after(1.0, lambda: stt.say("It's Sarah about a blood test")))
        outcome = await session.run()
        assert outcome.kind == "message"
        assert outcome.details["caller_name"] == "Sarah"
        assert [e["who"] for e in session.transcript] == ["assistant", "caller", "assistant"]
        assert len(session.latencies) == 1

    asyncio.run(go())


def test_barge_in_trims_to_what_was_heard():
    async def go():
        long_reply = "one two three four five six seven eight nine ten eleven twelve"
        agent = StubAgent([(long_reply, []), ("Sure.", [ToolCall("t2", "hang_up", {})])])
        session, stt, link = make_session(agent)
        asyncio.create_task(after(1.0, lambda: stt.say("Hi")))
        # Reply audio starts ~1.0s; interrupt ~1.0s into it (about 4 words heard).
        asyncio.create_task(after(2.1, lambda: stt.say("Sorry, wait")))
        outcome = await session.run()
        assert outcome.kind == "hung_up"
        heard = agent.interruptions[0].split()
        assert 2 <= len(heard) <= 7, heard
        assert heard == long_reply.split()[: len(heard)]

    asyncio.run(go())


def test_backchannel_does_not_interrupt():
    async def go():
        reply = "one two three four five six seven eight"
        agent = StubAgent([(reply, [ToolCall("t9", "hang_up", {})])])
        session, stt, _ = make_session(agent)
        asyncio.create_task(after(1.0, lambda: stt.say("Hi")))
        asyncio.create_task(after(1.8, lambda: stt.event("Update", "Mhmm")))
        asyncio.create_task(after(2.0, lambda: stt.event("EndOfTurn", "Mhmm.")))
        outcome = await session.run()
        assert outcome.kind == "hung_up"
        assert agent.interruptions == []
        assert len(agent.heard) == 1  # "Mhmm" wasn't sent to Claude as a turn
        assert reply in [e["text"] for e in session.transcript]

    asyncio.run(go())


def test_escalate_take_it():
    async def go():
        agent = StubAgent([("One moment.", [ToolCall("t3", "escalate", {"announcement": "Mum, urgent"})])])
        session, stt, _ = make_session(agent, decision="take")
        asyncio.create_task(after(1.0, lambda: stt.say("It's Mum, it's urgent")))
        outcome = await session.run()
        assert outcome.kind == "handed_off"

    asyncio.run(go())


def test_escalate_declined_takes_message():
    async def go():
        agent = StubAgent([
            ("One moment.", [ToolCall("t4", "escalate", {"announcement": "Plumber"})]),
            ("They can't come to the phone. What's the message?", []),
            ("Got it, bye.", [ToolCall("t5", "take_message", {"reason": "quote"})]),
        ])
        session, stt, _ = make_session(agent, decision="message")
        asyncio.create_task(after(1.0, lambda: stt.say("It's the plumber")))
        asyncio.create_task(after(4.0, lambda: stt.say("Tell him the quote is ready")))
        outcome = await session.run()
        assert outcome.kind == "message"
        assert agent.heard[1] == ("t4", "The person they're calling can't take the call right now. Let the caller know and take a message.")

    asyncio.run(go())


def test_caller_hangs_up():
    async def go():
        session, _, link = make_session(StubAgent([]))
        asyncio.create_task(after(1.0, link.close))
        assert (await session.run()).kind == "caller_hung_up"

    asyncio.run(go())


def test_eager_reply_released_on_end_of_turn():
    async def go():
        agent = StubAgent([("Thanks, goodbye.", [ToolCall("t6", "take_message", {"reason": "test"})])])
        session, stt, _ = make_session(agent)
        asyncio.create_task(after(1.0, lambda: stt.event("StartOfTurn")))
        asyncio.create_task(after(1.1, lambda: stt.event("EagerEndOfTurn", "It's Sarah.")))
        asyncio.create_task(after(1.6, lambda: stt.event("EndOfTurn", "it's sarah")))
        outcome = await session.run()
        assert outcome.kind == "message"
        assert len(agent.heard) == 1  # the speculative request was used, not repeated
        assert [e["who"] for e in session.transcript] == ["assistant", "caller", "assistant"]
        assert session.latencies[0]["eot_to_first_audio"] < 0.2  # audio was ready and waiting

    asyncio.run(go())


def test_eager_reply_discarded_when_caller_keeps_talking():
    async def go():
        agent = StubAgent([("Wrong guess.", []), ("Right answer, goodbye.", [ToolCall("t7", "hang_up", {})])])
        session, stt, _ = make_session(agent)
        asyncio.create_task(after(1.0, lambda: stt.event("StartOfTurn")))
        asyncio.create_task(after(1.1, lambda: stt.event("EagerEndOfTurn", "It's")))
        asyncio.create_task(after(1.4, lambda: stt.event("TurnResumed")))
        asyncio.create_task(after(2.0, lambda: stt.event("EndOfTurn", "It's Sarah about a test")))
        outcome = await session.run()
        assert outcome.kind == "hung_up"
        assert agent.discarded == 1
        assistant_lines = [e["text"] for e in session.transcript if e["who"] == "assistant"]
        assert "Wrong guess." not in assistant_lines and "Right answer, goodbye." in assistant_lines

    asyncio.run(go())


def test_eager_and_final_in_the_same_tick():
    """Flux often sends EagerEndOfTurn and EndOfTurn together; the right reply must play."""
    async def go():
        agent = StubAgent([("First reply.", []), ("Second reply, bye.", [ToolCall("t10", "hang_up", {})])])
        session, stt, _ = make_session(agent)

        def both(text):
            stt.event("EagerEndOfTurn", text)
            stt.event("EndOfTurn", text)

        asyncio.create_task(after(1.0, lambda: both("Hello")))
        asyncio.create_task(after(3.0, lambda: both("It's Bob")))
        outcome = await session.run()
        assert outcome.kind == "hung_up"
        assert agent.interruptions == []
        lines = [e["text"] for e in session.transcript if e["who"] == "assistant"]
        assert lines == ["Hello there.", "First reply.", "Second reply, bye."]

    asyncio.run(go())


def test_silent_hang_up_still_says_goodbye():
    async def go():
        agent = StubAgent([("", [ToolCall("t8", "hang_up", {"reason": "robocall"})])])
        session, stt, _ = make_session(agent)
        asyncio.create_task(after(1.0, lambda: stt.say("Press one for a special offer")))
        assert (await session.run()).kind == "hung_up"
        assert session.transcript[-1]["text"] == "Goodbye."

    asyncio.run(go())


def test_normalise():
    assert normalise("07700 900 456") == normalise("+447700900456") == normalise("00447700900456") == "+447700900456"
    assert normalise("") == ""


def test_numbers_match_on_last_digits():
    known = NumberSet({"07700 900789": "Jack", "+44 20 7946 0000": "Office", "121": "Voicemail"})
    assert known.lookup("+447700900789") == "Jack"
    assert known.lookup("00447700900789") == "Jack"
    assert known.lookup("07700900789") == "Jack"
    assert known.lookup("020 7946 0000") == "Office"
    assert known.lookup("07700900788") is None
    assert known.lookup("121") == "Voicemail" and known.lookup("0121") is None  # short codes match in full
    assert known.lookup("") is None


def test_parse_vcards():
    text = ("BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Jack Smith\r\nTEL;TYPE=CELL:+44 7700\r\n 900789\r\n"
            "TEL;TYPE=HOME:020 7946 0000\r\nEND:VCARD\r\n"
            "BEGIN:VCARD\r\nVERSION:3.0\r\nN:Doe;Jane;;;\r\nTEL:07700 900123\r\nEND:VCARD\r\n"
            "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:No Number\r\nEND:VCARD\r\n")
    assert parse_vcards(text) == [
        {"name": "Jack Smith", "numbers": ["+44 7700900789", "020 7946 0000"]},
        {"name": "Jane Doe", "numbers": ["07700 900123"]},
    ]
