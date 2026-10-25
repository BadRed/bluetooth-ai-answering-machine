#!/bin/sh
# Sync the working tree to the answering machine host. Usage: scripts/deploy.sh [user@host] (or set AM_HOST)
set -eu
HOST="${1:-${AM_HOST:?usage: scripts/deploy.sh user@host, or set AM_HOST}}"
cd "$(dirname "$0")/.."
rsync -az --delete \
    --exclude .git --exclude .venv --exclude __pycache__ --exclude data --exclude .env \
    ./ "$HOST:/opt/answering-machine/"
