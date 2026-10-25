"""Phone number normalisation and matching. Caller ID from the iPhone arrives in national format (07…)."""

import re

MATCH_DIGITS = 9  # compare the last 9 digits, so 07…, +447… and 00447… all match


def normalise(number: str) -> str:
    """Return a UK-centric E.164-ish form so 07…, +447… and 00447… compare equal."""
    digits = re.sub(r"[^\d+]", "", number or "")
    if digits.startswith("00"):
        digits = "+" + digits[2:]
    elif digits.startswith("0"):
        digits = "+44" + digits[1:]
    return digits


def match_key(number: str) -> str:
    """Key for 'is this the same number?': the last few digits, ignoring prefixes and spacing.

    Short numbers (under 7 digits, e.g. service codes) must match in full.
    """
    digits = re.sub(r"\D", "", number or "")
    return digits[-MATCH_DIGITS:] if len(digits) >= 7 else digits


class NumberSet:
    """Numbers to recognise, matched on their last digits. Maps each key to a label (e.g. a contact name)."""

    def __init__(self, numbers: dict[str, str] | None = None):
        self._keys: dict[str, str] = {}
        for number, label in (numbers or {}).items():
            self.add(number, label)

    def add(self, number: str, label: str = "") -> None:
        if key := match_key(number):
            self._keys[key] = label

    def lookup(self, number: str) -> str | None:
        """The label if the number is known (may be ""), else None."""
        key = match_key(number)
        return self._keys.get(key) if key else None

    def __len__(self) -> int:
        return len(self._keys)
