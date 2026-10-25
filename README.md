# Bluetooth AI Answering Machine

A modern answering machine. It pairs with your phone as a Bluetooth hands-free unit (like a car kit) and lets an AI
voice agent screen your calls. It works with iPhone and Android, with no app and no carrier involvement.

- **Unknown callers** hear a natural "Hello?". The assistant finds out who they are and why they're calling. Then it takes a
  message, hangs up on spam and scams, or puts important callers on hold and asks you on Telegram.
- **Take it:** tap the Telegram button and the call moves to your phone's handset.
- **Known callers** (your phone's contacts, synced over Bluetooth, or an allowlist) ring through as normal.
- After every screened call you get a short summary and transcript on Telegram.

## How it works

```
 Phone ──Bluetooth HFP──▶ Linux box ──WebSocket──▶ Deepgram Voice Agent
 (call control + 8 kHz    BlueZ + oFono            Flux speech-to-text, Claude Haiku 4.5,
  call audio)             answering_machine        Cartesia (or Deepgram Aura) voice
                                │
                                └──▶ Telegram bot: escalations, summaries, /status /pause /resume
```

- **Call control:** oFono, over D-Bus, in the HFP Hands-Free role: caller ID, answer, hang up, dial.
- **Call audio:** CVSD, 8 kHz, 16-bit mono PCM, passed straight to the voice agent with no resampling.
- **Audio ownership:** the Bluetooth control link stays connected, but the box only holds the call-audio link (SCO) for calls it
  answered. For calls you take or make yourself, it refuses the audio link, so audio stays on your phone. When it isn't running,
  oFono refuses the audio link on its own.

## Hardware

You need a Linux machine whose Bluetooth controller carries call audio (SCO) over HCI. **USB Bluetooth controllers do; many
onboard ones don't.**

- **Tested:** Intel N100 mini PC with its built-in Realtek RTL8821CE (a USB-attached controller), Debian 12, iPhone.
- **Raspberry Pi onboard Bluetooth** (Zero W, Zero 2 W, 3, 4) routes call audio to the PCM pins by default. Use a USB dongle
  instead (e.g. one with a Realtek RTL8761B chip).
- **Range:** keep the box in the same room as your phone. Antennas inside a small metal case don't reach far. A USB dongle on
  an extension cable helps.

## Setup

On the Linux box, as root:

```sh
apt install bluez ofono bluez-obexd          # no PulseAudio/PipeWire: they'd grab the hands-free profile
systemctl enable --now bluetooth ofono
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/BadRed/bluetooth-ai-answering-machine /opt/answering-machine
cd /opt/answering-machine && cp .env.example .env && chmod 600 .env    # then fill it in
```

1. **Pair your phone:** in `bluetoothctl`, run `agent on`, `default-agent`, `discoverable on` and `pairable on`. Pair from the
   phone, confirm the passkey, then run `trust <phone address>`. On Android, turn on **Phone calls** for the device.
2. **Keys:** you need [Deepgram](https://deepgram.com) (the voice agent) and [Cartesia](https://cartesia.ai) (the voice; or set
   `DG_SPEAK_PROVIDER=deepgram`). Anthropic is optional and only used for call summaries.
3. **Telegram:** create a bot with @BotFather, put the token in `.env`, send the bot a message, then run
   `uv run --env-file .env -m answering_machine --telegram-whoami` and copy your chat ID into `.env`.
4. **Contacts (optional):** iPhone: Settings → Bluetooth → (i) next to the box → **Sync Contacts**. Then run
   `scripts/sync-contacts.sh`. Contacts are stored only on the box (`data/contacts.json`, mode 600) and are used only to decide
   whether a caller is known.

## Running

```sh
uv run --env-file .env -m answering_machine                     # screen calls on the real phone
uv run --env-file .env -m answering_machine --screen-all        # testing: answer every call, contacts included
uv run --env-file .env -m answering_machine --fake assets/callers/gp --console   # scripted caller, no phone needed
uv run --env-file .env --extra mic -m answering_machine --fake mic             # talk to it yourself (Mac/Linux mic)
uv run spike.py --help                                          # low-level Bluetooth/audio test tool
uv run pytest
```

Scripted callers live in `assets/callers/<scenario>/NN.wav` (8 kHz mono). Each clip plays once the assistant goes quiet, and
the caller's perceived response time is logged. Every call is saved to `data/calls/*.json` (transcript, outcome, latency).

`--engine local` runs an alternative voice loop built from parts (Deepgram Flux + Claude via the Anthropic API + Cartesia),
wired together in `session.py`. It has its own turn-taking, barge-in and backchannel handling.

## Privacy and behaviour

- The assistant answers with just "Hello?" and **never says your name until the caller has**. If they ask for you by name, it
  says it's your assistant.
- It won't share anything about you (whereabouts, schedule, contact details) and treats urgent-sounding unknown callers with
  suspicion.
- It doesn't announce that it's an AI, but **if a caller sincerely asks, it tells them**.
- Call records and contacts stay on the box. Audio is only recorded in `--fake` test mode.
- You're responsible for how you use it. In the UK, automated *marketing* calls fall under PECR. This project screens calls
  you receive.

## Things we learned the hard way

- **oFono hands the audio agent a deferred-setup SCO socket.** The kernel only accepts the phone's audio connection once you
  `read()` from it. If you wait for it to become readable first, nothing happens until the phone gives up about 20 s later.
- oFono's introspection data doesn't match what `HandsfreeAudioManager.GetCards` actually returns, so call it with a raw D-Bus
  message.
- iPhones open the audio link while ringing (in-band ringtone). Refuse it unless you're answering.
- The iPhone doesn't send contact names with caller ID, hence the contact sync over Bluetooth.

## Costs

Roughly $0.075 per minute of screened call for the Deepgram Voice Agent (pay as you go), plus Cartesia characters if you use a
Cartesia voice. Summaries cost a fraction of a cent per call.

## Status

Working: screening real calls end to end, Telegram escalation and summaries, contact sync, scripted test callers.
Next: run as a service with auto-reconnect and fail-safes, then more Telegram control, then outbound calls (dial, navigate
phone menus, wait on hold).
