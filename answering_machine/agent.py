"""Claude side of a screening call: streams replies into TTS and returns any tool calls."""

import logging
from dataclasses import dataclass

import anthropic

from answering_machine.prompts import SUMMARY_SYSTEM, TOOLS, screening_system
from answering_machine.tts import Utterance

log = logging.getLogger("answering_machine.agent")


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


def _valid(name: str, args: object) -> bool:
    """Eager input streaming skips server-side validation, so check inputs ourselves."""
    if not isinstance(args, dict):
        return False
    if name == "escalate":
        return isinstance(args.get("announcement"), str) and bool(args["announcement"].strip())
    if name == "take_message":
        return isinstance(args.get("reason"), str) and all(isinstance(v, str) for v in args.values())
    if name == "hang_up":
        return all(isinstance(v, str) for v in args.values())
    return False


class ScreeningAgent:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str, owner: str, caller_number: str):
        self._client = client
        self._model = model
        self._system = [{"type": "text", "text": screening_system(owner), "cache_control": {"type": "ephemeral"}}]
        self._caller = caller_number or "withheld"
        self.messages: list[dict] = []

    async def respond(self, utt: Utterance, text: str | None = None, tool_result: tuple[str, str] | None = None) -> list[ToolCall]:
        """Stream one assistant turn into `utt`. Cancelling mid-stream records nothing; call interrupted()."""
        if tool_result:
            tool_id, result = tool_result
            self.messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": result}]})
        else:
            if not self.messages:
                text = f"[Caller ID: {self._caller}]\n{text}"
            self.messages.append({"role": "user", "content": text})

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=self._system,
            tools=TOOLS,
            messages=self.messages,
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    await utt.send(event.text)
            final = await stream.get_final_message()
        await utt.finish()

        self.messages.append({"role": "assistant", "content": [b.to_dict() for b in final.content]})
        if final.stop_reason in ("max_tokens", "refusal"):
            log.warning("Reply stopped with %s; ignoring any tool call", final.stop_reason)
            return []

        calls = []
        for block in final.content:
            if block.type != "tool_use":
                continue
            if _valid(block.name, block.input):
                calls.append(ToolCall(block.id, block.name, block.input))
            else:
                log.warning("Invalid %s input: %r", block.name, block.input)
        return calls

    def discard_pending(self) -> None:
        """Forget a speculative turn that was thrown away before the caller heard any of it."""
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        if self.messages and self.messages[-1]["role"] == "user" and isinstance(self.messages[-1]["content"], str):
            self.messages.pop()

    def interrupted(self, spoken: str) -> None:
        """Make history reflect only what the caller heard before they cut in."""
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()  # finished streaming but was cut off during playback; drop its tool calls too
        if spoken:
            self.messages.append({"role": "assistant", "content": spoken + " —"})


async def summarise(client: anthropic.AsyncAnthropic, model: str, transcript: str, outcome: str) -> str:
    try:
        response = await client.beta.messages.create(
            model=model,
            max_tokens=2048,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            system=SUMMARY_SYSTEM,
            messages=[{"role": "user", "content": f"Outcome: {outcome}\n\nTranscript:\n{transcript}"}],
        )
    except anthropic.APIError as e:
        log.error("Summary failed: %s", e)
        return "(summary unavailable)"
    if response.stop_reason == "refusal":
        return "(summary unavailable)"
    return "".join(b.text for b in response.content if b.type == "text").strip()
