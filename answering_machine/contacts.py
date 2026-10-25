"""Contacts from the phone over Bluetooth (PBAP), kept only on this box.

Pulled into data/contacts.json (root-only) and used solely to decide whether a caller is
known, so known callers ring through unscreened. Never sent anywhere.

Run on the host via scripts/sync-contacts.sh, which starts obexd on a private D-Bus session.
The phone must allow it: iPhone → Settings → Bluetooth → (i) next to this box → Sync Contacts.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from dbus_fast import BusType, Message, MessageType, Variant
from dbus_fast.aio import MessageBus

from answering_machine.numbers import NumberSet

log = logging.getLogger("answering_machine.contacts")

OBEX = "org.bluez.obex"
HANDSFREE_AG_UUID = "0000111f-0000-1000-8000-00805f9b34fb"


def contacts_path(data_dir: str) -> Path:
    return Path(data_dir) / "contacts.json"


def load(data_dir: str) -> NumberSet:
    """Known numbers from the last sync (empty if never synced)."""
    path = contacts_path(data_dir)
    if not path.exists():
        return NumberSet()
    known = NumberSet()
    for contact in json.loads(path.read_text())["contacts"]:
        for number in contact["numbers"]:
            known.add(number, contact["name"])
    return known


def parse_vcards(text: str) -> list[dict]:
    """Minimal vCard 3.0 parsing: name and phone numbers per card."""
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]  # folded continuation line
        else:
            lines.append(raw)

    contacts, card = [], None
    for line in lines:
        key, _, value = line.partition(":")
        prop = key.split(";")[0].upper()
        if prop == "BEGIN":
            card = {"name": "", "numbers": []}
        elif prop == "END" and card is not None:
            if card["numbers"]:
                contacts.append(card)
            card = None
        elif card is None:
            continue
        elif prop == "FN":
            card["name"] = value.strip()
        elif prop == "N" and not card["name"]:
            card["name"] = " ".join(p for p in reversed(value.split(";")[:2]) if p).strip()
        elif prop == "TEL" and value.strip():
            card["numbers"].append(value.strip())
    return contacts


async def _call(bus: MessageBus, destination: str, path: str, interface: str, member: str,
                signature: str = "", body: list | None = None) -> Message:
    reply = await bus.call(Message(destination=destination, path=path, interface=interface, member=member,
                                   signature=signature, body=body or []))
    if reply.message_type == MessageType.ERROR:
        raise RuntimeError(f"{interface}.{member}: {reply.error_name} {reply.body}")
    return reply


async def _phone_address() -> str:
    """Bluetooth address of the connected phone (the device offering hands-free audio gateway)."""
    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    reply = await _call(bus, "org.bluez", "/", "org.freedesktop.DBus.ObjectManager", "GetManagedObjects")
    bus.disconnect()
    for path, interfaces in reply.body[0].items():
        dev = interfaces.get("org.bluez.Device1")
        if dev and dev["Connected"].value and HANDSFREE_AG_UUID in dev["UUIDs"].value:
            return dev["Address"].value
    raise RuntimeError("No connected phone found")


async def pull(address: str) -> str:
    """Download the phone's main phonebook as vCard text."""
    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    session = (await _call(bus, OBEX, "/org/bluez/obex", "org.bluez.obex.Client1", "CreateSession",
                           "sa{sv}", [address, {"Target": Variant("s", "pbap")}])).body[0]
    try:
        await _call(bus, OBEX, session, "org.bluez.obex.PhonebookAccess1", "Select", "ss", ["int", "pb"])
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "pb.vcf")
            filters = {"Format": Variant("s", "vcard30"), "Fields": Variant("as", ["VERSION", "FN", "N", "TEL"])}
            transfer = (await _call(bus, OBEX, session, "org.bluez.obex.PhonebookAccess1", "PullAll",
                                    "sa{sv}", [target, filters])).body[0]
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                try:
                    status = (await _call(bus, OBEX, transfer, "org.freedesktop.DBus.Properties", "Get", "ss",
                                          ["org.bluez.obex.Transfer1", "Status"])).body[0].value
                except RuntimeError:
                    status = "complete" if os.path.exists(target) else "gone"  # finished transfers vanish
                if status == "complete":
                    return Path(target).read_text(errors="replace")
                if status in ("error", "gone"):
                    raise RuntimeError(f"Phonebook transfer {status}")
                await asyncio.sleep(0.5)
            raise RuntimeError("Phonebook transfer timed out")
    finally:
        await _call(bus, OBEX, "/org/bluez/obex", "org.bluez.obex.Client1", "RemoveSession", "o", [session])
        bus.disconnect()


async def sync(data_dir: str) -> int:
    address = await _phone_address()
    log.info("Pulling contacts from %s", address)
    contacts = parse_vcards(await pull(address))
    path = contacts_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"synced": time.strftime("%Y-%m-%d %H:%M:%S"), "contacts": contacts}, indent=1))
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    log.info("Saved %d contacts with numbers to %s", len(contacts), path)
    return len(contacts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    try:
        asyncio.run(sync(os.environ.get("DATA_DIR", "data")))
    except RuntimeError as e:
        sys.exit(f"Contact sync failed: {e}")
