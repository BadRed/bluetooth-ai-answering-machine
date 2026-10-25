"""Settings from the environment. Run with `uv run --env-file .env ...`."""

import os
from dataclasses import dataclass, field

from answering_machine.numbers import normalise


def _list(name: str) -> list[str]:
    return [normalise(n) for n in os.environ.get(name, "").split(",") if n.strip()]


@dataclass(frozen=True)
class Config:
    engine: str = field(default_factory=lambda: os.environ.get("ENGINE", "deepgram"))  # deepgram | local
    dg_think_provider: str = field(default_factory=lambda: os.environ.get("DG_THINK_PROVIDER", "anthropic"))
    dg_think_model: str = field(default_factory=lambda: os.environ.get("DG_THINK_MODEL", "claude-haiku-4-5"))
    dg_speak_provider: str = field(default_factory=lambda: os.environ.get("DG_SPEAK_PROVIDER", "cartesia"))  # cartesia | deepgram
    dg_voice: str = field(default_factory=lambda: os.environ.get("DG_VOICE", "aura-2-pandora-en"))
    owner_name: str = field(default_factory=lambda: os.environ.get("OWNER_NAME", ""))
    anthropic_model: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5"))
    summary_model: str = field(default_factory=lambda: os.environ.get("SUMMARY_MODEL", "claude-sonnet-5-5"))
    deepgram_api_key: str = field(default_factory=lambda: os.environ.get("DEEPGRAM_API_KEY", ""))
    cartesia_api_key: str = field(default_factory=lambda: os.environ.get("CARTESIA_API_KEY", ""))
    cartesia_voice_id: str = field(default_factory=lambda: os.environ.get("CARTESIA_VOICE_ID", ""))
    cartesia_model: str = field(default_factory=lambda: os.environ.get("CARTESIA_MODEL", "sonic-3.6"))
    telegram_bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))
    screen_all: bool = field(default_factory=lambda: os.environ.get("SCREEN_ALL", "") == "1")  # testing: ignore contacts/allowlist
    allowlist: list[str] = field(default_factory=lambda: _list("ALLOWLIST"))
    ring_delay: float = field(default_factory=lambda: float(os.environ.get("RING_DELAY", "8")))
    escalation_timeout: float = field(default_factory=lambda: float(os.environ.get("ESCALATION_TIMEOUT", "45")))
    max_call_seconds: float = field(default_factory=lambda: float(os.environ.get("MAX_CALL_SECONDS", "300")))
    data_dir: str = field(default_factory=lambda: os.environ.get("DATA_DIR", "data"))

    def missing(self, *names: str) -> list[str]:
        return [n.upper() for n in names if not getattr(self, n)]
