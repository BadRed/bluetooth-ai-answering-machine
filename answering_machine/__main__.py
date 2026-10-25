"""Run the Bluetooth AI answering machine.

  uv run --env-file .env -m answering_machine                      # real phone (on the host), screening
  uv run --env-file .env -m answering_machine --echo               # real phone, echo bot
  uv run --env-file .env -m answering_machine --fake assets/callers/gp     # scripted caller, no Bluetooth
  uv run --env-file .env --extra mic -m answering_machine --fake mic       # talk to it from your Mac
  uv run --env-file .env -m answering_machine --telegram-whoami    # print chat IDs that messaged the bot
"""

import argparse
import asyncio
import dataclasses
import logging
import sys
import time
from pathlib import Path

from answering_machine.config import Config


def build_fake_phone(spec: str, number: str, data_dir: str):
    from answering_machine.fakephone import FakePhone, MicSource, ScriptSource, SpeakerSink, StereoWavSink, TeeSink

    recording = StereoWavSink(Path(data_dir) / "recordings" / time.strftime("%Y%m%d-%H%M%S-fake.wav"))
    if spec == "mic":
        return FakePhone(MicSource(), TeeSink(recording, SpeakerSink()), number), recording.path
    clips = sorted(Path(spec).glob("*.wav"))
    if not clips:
        sys.exit(f"No .wav caller clips in {spec}")
    return FakePhone(ScriptSource(clips), recording, number), recording.path


async def telegram_whoami(cfg: Config) -> None:
    import httpx

    async with httpx.AsyncClient() as http:
        r = await http.get(f"https://api.telegram.org/bot{cfg.telegram_bot_token}/getUpdates")
    chats = {u["message"]["chat"]["id"]: u["message"]["chat"].get("username") for u in r.json().get("result", []) if "message" in u}
    print("\n".join(f"{cid}  @{name}" for cid, name in chats.items()) or "No messages yet: send your bot any message first.")


async def main_async(args: argparse.Namespace) -> None:
    from answering_machine.app import AnsweringMachine
    from answering_machine.notify import Console, Telegram

    cfg = Config()
    if args.telegram_whoami:
        return await telegram_whoami(cfg)

    if args.engine:
        cfg = dataclasses.replace(cfg, engine=args.engine)
    if args.screen_all:
        cfg = dataclasses.replace(cfg, screen_all=True)
    needs = ["deepgram_api_key", "owner_name"]
    if cfg.engine == "local" or args.echo or cfg.dg_speak_provider == "cartesia":
        needs += ["cartesia_api_key", "cartesia_voice_id"]
    missing = cfg.missing(*needs)
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")

    use_telegram = cfg.telegram_bot_token and cfg.telegram_chat_id and not args.console
    notifier = Telegram(cfg.telegram_bot_token, cfg.telegram_chat_id) if use_telegram else Console()

    if args.fake:
        cfg = dataclasses.replace(cfg, ring_delay=1.0)
        phone, recording = build_fake_phone(args.fake, args.number, cfg.data_dir)
        machine = AnsweringMachine(cfg, phone, notifier, echo=args.echo)
        await phone.start()
        notify_task = asyncio.create_task(notifier.run())
        await phone.ended.wait()
        await machine.wait_idle()
        notify_task.cancel()
        logging.getLogger("answering_machine").info("Recording (L caller, R assistant): %s", recording)
        return

    from answering_machine.phone import Phone

    phone = Phone()
    machine = AnsweringMachine(cfg, phone, notifier, echo=args.echo)
    try:
        await machine.run()
    finally:
        phone.release_audio()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--echo", action="store_true", help="echo bot instead of the screening agent (local engine)")
    parser.add_argument("--engine", choices=["deepgram", "local"], help="voice loop: Deepgram Voice Agent, or our own")
    parser.add_argument("--fake", metavar="mic|DIR", help="simulated call: 'mic', or a directory of caller .wav clips")
    parser.add_argument("--number", default="07700900123", help="caller ID for --fake")
    parser.add_argument("--console", action="store_true", help="use the console instead of Telegram")
    parser.add_argument("--screen-all", action="store_true", help="testing: answer every call, contacts and allowlist included")
    parser.add_argument("--telegram-whoami", action="store_true", help="print chat IDs that have messaged the bot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
