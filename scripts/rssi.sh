#!/bin/sh
# Log the phone's Bluetooth signal strength every 2 s (run on the host). Usage: scripts/rssi.sh MAC
# Rough guide for call audio: above -70 dBm fine, -70..-80 crackly, below -85 drops.
MAC="${1:?usage: scripts/rssi.sh <phone Bluetooth address, see: bluetoothctl devices>}"
while true; do
    printf '%s ' "$(date +%T)"
    btmgmt conn-info "$MAC" 2>&1 | grep -oE 'RSSI -?[0-9]+' || echo "not connected"
    sleep 2
done
