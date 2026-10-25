#!/bin/sh
# Pull the phone's contacts over Bluetooth (PBAP) into data/contacts.json. Run on the host as root.
# obexd lives on a per-user D-Bus session, so start a private one just for this.
set -eu
cd "$(dirname "$0")/.."
exec dbus-run-session -- sh -c '
    /usr/libexec/bluetooth/obexd -n & obexd=$!
    sleep 1
    uv run --env-file .env python -m answering_machine.contacts; status=$?
    kill $obexd
    exit $status
'
