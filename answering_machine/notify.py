"""How the answering machine reaches the user: Telegram, or the console when testing without it."""

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable

import httpx

log = logging.getLogger("answering_machine.notify")

Options = list[tuple[str, str]]  # (button label, value)
CommandHandler = Callable[[str, str], Awaitable[str]]  # (command, args) -> reply


class Telegram:
    """Minimal Bot API client: long polling, messages, inline buttons. Only `chat_id` is served."""

    def __init__(self, token: str, chat_id: str):
        self._api = f"https://api.telegram.org/bot{token}"
        self._chat_id = str(chat_id)
        self._http = httpx.AsyncClient(timeout=70)
        self._pending: dict[int, asyncio.Future] = {}  # Telegram message_id of the question -> answer
        self.on_command: CommandHandler | None = None

    async def notify(self, text: str) -> None:
        await self._call("sendMessage", chat_id=self._chat_id, text=text)

    async def ask(self, text: str, options: Options, timeout: float) -> str | None:
        """Send a message with buttons and wait for a tap; None on timeout."""
        keyboard = [[{"text": label, "callback_data": value}] for label, value in options]
        msg = await self._call("sendMessage", chat_id=self._chat_id, text=text, reply_markup={"inline_keyboard": keyboard})
        future = asyncio.get_running_loop().create_future()
        self._pending[msg["message_id"]] = future  # taps are matched to this exact message
        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            return None
        finally:
            self._pending.pop(msg["message_id"], None)
            try:  # remove the buttons so stale taps aren't possible
                await self._call("editMessageReplyMarkup", chat_id=self._chat_id, message_id=msg["message_id"],
                                 reply_markup={"inline_keyboard": []})
            except (httpx.HTTPError, RuntimeError) as e:
                log.warning("Couldn't clear buttons: %s", e)

    async def run(self) -> None:
        # Taps and messages sent while the answering machine wasn't running are stale: drop them.
        await self._call("deleteWebhook", drop_pending_updates=True)
        offset = 0
        while True:
            try:
                updates = await self._call("getUpdates", offset=offset, timeout=50,
                                           allowed_updates=["message", "callback_query"])
            except (httpx.HTTPError, RuntimeError) as e:
                log.warning("Telegram poll failed: %s", e)
                await asyncio.sleep(5)
                continue
            for update in updates:
                offset = update["update_id"] + 1
                await self._handle(update)

    async def _handle(self, update: dict) -> None:
        if query := update.get("callback_query"):
            if str(query["message"]["chat"]["id"]) != self._chat_id:
                return
            value = query.get("data", "")
            future = self._pending.get(query["message"]["message_id"])
            log.info("Button %r tapped on message %s (waiting on %s)", value, query["message"]["message_id"],
                     list(self._pending) or "none")
            answered = future is not None and not future.done()
            if answered:
                future.set_result(value)
            await self._call("answerCallbackQuery", callback_query_id=query["id"],
                             text="Done" if answered else "Too late, that call has moved on")
        elif msg := update.get("message"):
            if str(msg["chat"]["id"]) != self._chat_id:
                log.warning("Ignoring message from chat %s", msg["chat"]["id"])
                return
            text = msg.get("text", "")
            if text.startswith("/") and self.on_command:
                cmd, _, args = text[1:].partition(" ")
                await self.notify(await self.on_command(cmd.split("@")[0].lower(), args.strip()))

    async def _call(self, method: str, **params) -> object:
        r = await self._http.post(f"{self._api}/{method}", json=params)
        body = r.json()
        if not body.get("ok"):
            raise RuntimeError(f"Telegram {method}: {body.get('description')}")
        return body["result"]


class Console:
    """Stand-in for Telegram when testing: prints notifications, asks on stdin."""

    on_command: CommandHandler | None = None

    async def notify(self, text: str) -> None:
        print(f"\n[notify] {text}\n", flush=True)

    async def ask(self, text: str, options: Options, timeout: float) -> str | None:
        choices = " / ".join(f"[{i}] {label}" for i, (label, _) in enumerate(options, 1))
        print(f"\n[ask] {text}\n      {choices}  (timeout {timeout:.0f}s)", flush=True)
        loop = asyncio.get_running_loop()
        try:
            line = await asyncio.wait_for(loop.run_in_executor(None, sys.stdin.readline), timeout)
        except TimeoutError:
            return None
        try:
            return options[int(line.strip()) - 1][1]
        except (ValueError, IndexError):
            return None

    async def run(self) -> None:
        await asyncio.Event().wait()
