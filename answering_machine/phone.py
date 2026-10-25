"""oFono D-Bus wrapper: brings up the phone's HFP modem, exposes call events and control,
and owns the SCO audio link.

Audio ownership rule: oFono hands every new SCO connection to our agent, and we close any
that `claim_audio()` doesn't want, so the phone falls back to its own earpiece.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Annotated

from dbus_fast import BusType, DBusError, Message, MessageType, Variant
from dbus_fast.aio import MessageBus, ProxyInterface
from dbus_fast.annotations import DBusSignature
from dbus_fast.service import ServiceInterface, dbus_method

log = logging.getLogger("answering_machine.phone")

OFONO = "org.ofono"
AGENT_PATH = "/answering_machine/audio_agent"
CODEC_CVSD = 0x01

ObjectPath = Annotated[str, DBusSignature("o")]
UnixFd = Annotated[int, DBusSignature("h")]
Byte = Annotated[int, DBusSignature("y")]


class ScoLink:
    """One SCO connection carrying CVSD audio as 8 kHz s16le mono PCM.

    Output is paced off the input: for every packet the phone sends, we send one packet
    of the same size back, padded with silence when nothing is queued.
    """

    SAMPLE_RATE = 8000

    def __init__(self, fd: int):
        self.fd = fd
        self.on_audio: Callable[[bytes], None] = lambda pcm: None
        self.closed = asyncio.Event()
        self._out = bytearray()
        os.set_blocking(fd, False)
        # oFono's SCO sockets use deferred setup: the kernel only accepts the phone's audio
        # connection request on our first read. Until then the socket never becomes readable,
        # so waiting for it would deadlock until the phone gives up (~20 s).
        try:
            os.read(fd, 1024)
        except BlockingIOError:
            pass
        asyncio.get_running_loop().add_reader(fd, self._readable)

    def play(self, pcm: bytes) -> None:
        self._out += pcm

    def clear_output(self) -> None:
        self._out.clear()

    @property
    def queued(self) -> int:
        """Bytes of output not yet sent to the phone."""
        return len(self._out)

    def close(self) -> None:
        if self.closed.is_set():
            return
        asyncio.get_running_loop().remove_reader(self.fd)
        os.close(self.fd)
        self.closed.set()
        log.info("SCO link closed")

    def _readable(self) -> None:
        try:
            packet = os.read(self.fd, 1024)
        except BlockingIOError:
            return
        except OSError as e:
            log.info("SCO read failed: %s", e)
            self.close()
            return
        if not packet:
            self.close()
            return
        self.on_audio(packet)

        n = len(packet)
        chunk = bytes(self._out[:n]).ljust(n, b"\0")
        del self._out[:n]
        try:
            os.write(self.fd, chunk)
        except BlockingIOError:
            pass
        except OSError as e:
            log.info("SCO write failed: %s", e)
            self.close()


class Call:
    def __init__(self, path: str, props: dict[str, Variant], iface: ProxyInterface):
        self.path = path
        self.number: str = props["LineIdentification"].value
        self.name: str = props["Name"].value if "Name" in props else ""  # contact name, if the phone sends one
        self.state: str = props["State"].value
        self._iface = iface

    async def answer(self) -> None:
        await self._iface.call_answer()

    async def hangup(self) -> None:
        await self._iface.call_hangup()

    def __repr__(self) -> str:
        who = f"{self.number or 'withheld'}" + (f" ({self.name})" if self.name else "")
        return f"<Call {self.path.rsplit('/', 1)[-1]} {who} {self.state}>"


class _AudioAgent(ServiceInterface):
    def __init__(self, phone: "Phone"):
        super().__init__("org.ofono.HandsfreeAudioAgent")
        self._phone = phone

    @dbus_method()
    def NewConnection(self, card: ObjectPath, fd: UnixFd, codec: Byte) -> None:
        self._phone._new_sco(card, fd, codec)

    @dbus_method()
    def Release(self) -> None:
        log.warning("oFono released our audio agent")


class Phone:
    def __init__(self) -> None:
        self.calls: dict[str, Call] = {}
        self.link: ScoLink | None = None

        # Set by the app.
        self.claim_audio: Callable[[], bool] = lambda: False
        self.on_call_added: Callable[[Call], None] = lambda call: None
        self.on_call_state: Callable[[Call], None] = lambda call: None
        self.on_call_removed: Callable[[Call], None] = lambda call: None
        self.on_audio_open: Callable[[ScoLink], None] = lambda link: None

        self._bus: MessageBus | None = None
        self._vcm: ProxyInterface | None = None
        self._audio_manager: ProxyInterface | None = None
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self._bus = await MessageBus(bus_type=BusType.SYSTEM, negotiate_unix_fd=True).connect()

        root = await self._proxy("/")
        manager = root.get_interface("org.ofono.Manager")
        manager.on_modem_added(lambda path, props: self._spawn(self._setup_modem(path)))
        manager.on_modem_removed(self._modem_removed)

        self._bus.export(AGENT_PATH, _AudioAgent(self))
        self._audio_manager = root.get_interface("org.ofono.HandsfreeAudioManager")
        await self._audio_manager.call_register(AGENT_PATH, bytes([CODEC_CVSD]))
        log.info("Audio agent registered (CVSD only)")

        modems = [path for path, _ in await manager.call_get_modems() if "/hfp/" in path]
        if not modems:
            log.warning("No HFP modem yet; waiting for the phone to connect")
        for path in modems:
            await self._setup_modem(path)

    @property
    def connected(self) -> bool:
        return self._vcm is not None

    async def dial(self, number: str) -> str:
        return await self._require_vcm().call_dial(number, "default")

    async def hangup_all(self) -> None:
        await self._require_vcm().call_hangup_all()

    async def send_tones(self, digits: str) -> None:
        await self._require_vcm().call_send_tones(digits)

    async def connect_audio(self) -> None:
        """Ask the phone to open SCO, for when it doesn't do so by itself."""
        # Raw calls: oFono's introspection data doesn't match what GetCards actually returns.
        reply = await self._raw_call("/", "org.ofono.HandsfreeAudioManager", "GetCards")
        for path, _ in reply.body[0]:
            await self._raw_call(path, "org.ofono.HandsfreeAudioCard", "Connect")
            return
        log.warning("No hands-free audio card to connect")

    async def _raw_call(self, path: str, interface: str, member: str) -> Message:
        reply = await self._bus.call(Message(destination=OFONO, path=path, interface=interface, member=member))
        if reply.message_type == MessageType.ERROR:
            raise DBusError(reply.error_name, str(reply.body))
        return reply

    def release_audio(self) -> None:
        if self.link:
            self.link.close()

    async def _proxy(self, path: str):
        introspection = await self._bus.introspect(OFONO, path)
        return self._bus.get_proxy_object(OFONO, path, introspection)

    def _require_vcm(self) -> ProxyInterface:
        if self._vcm is None:
            raise RuntimeError("Phone not connected")
        return self._vcm

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _setup_modem(self, path: str) -> None:
        if "/hfp/" not in path:
            return
        modem = (await self._proxy(path)).get_interface("org.ofono.Modem")
        for prop in ("Powered", "Online"):
            if not await self._enable(modem, prop):
                log.error("Couldn't set %s on %s", prop, path)
                return
        for _ in range(40):
            props = await modem.call_get_properties()
            if "org.ofono.VoiceCallManager" in props["Interfaces"].value:
                break
            await asyncio.sleep(0.25)
        else:
            log.error("Modem %s never exposed VoiceCallManager", path)
            return

        self._vcm = (await self._proxy(path)).get_interface("org.ofono.VoiceCallManager")
        self._vcm.on_call_added(lambda call_path, props: self._spawn(self._add_call(call_path, props)))
        self._vcm.on_call_removed(self._remove_call)
        log.info("Phone ready: %s", path)
        for call_path, props in await self._vcm.call_get_calls():
            await self._add_call(call_path, props)

    @staticmethod
    async def _enable(modem: ProxyInterface, prop: str, timeout: float = 90) -> bool:
        """Set a boolean modem property and wait for it to stick.

        Powering an HFP modem connects to the phone, which can take tens of seconds.
        oFono holds the SetProperty reply until it's done, so we send it once and wait;
        re-sending while it's pending only keeps the link busy.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if (await modem.call_get_properties())[prop].value:
                return True
            try:
                await modem.call_set_property(prop, Variant("b", True))
            except DBusError as e:
                log.info("Setting %s: %s; retrying", prop, e.text)
                await asyncio.sleep(3)
        return False

    def _modem_removed(self, path: str) -> None:
        if "/hfp/" in path:
            self._vcm = None
            log.warning("Phone disconnected: %s", path)

    async def _add_call(self, path: str, props: dict[str, Variant]) -> None:
        iface = (await self._proxy(path)).get_interface("org.ofono.VoiceCall")
        call = Call(path, props, iface)
        self.calls[path] = call

        def changed(name: str, value: Variant) -> None:
            if name == "State":
                call.state = value.value
                self.on_call_state(call)
            elif name == "LineIdentification":
                call.number = value.value
            elif name == "Name":
                call.name = value.value

        iface.on_property_changed(changed)
        self.on_call_added(call)

    def _remove_call(self, path: str) -> None:
        call = self.calls.pop(path, None)
        if call:
            self.on_call_removed(call)

    def _new_sco(self, card: str, fd: int, codec: int) -> None:
        if codec != CODEC_CVSD or not self.claim_audio():
            log.info("Refusing SCO (codec=%d) so audio stays on the phone", codec)
            os.close(fd)
            return
        if self.link:
            self.link.close()
        self.link = ScoLink(fd)
        log.info("SCO link open")
        self.on_audio_open(self.link)
