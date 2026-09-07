#!/usr/bin/env bash
#
# End to end, on a synthetic page, with no API key and no photograph.
#
# Synthesises a page with known skew, perspective and a strikethrough; runs it
# through preprocessing, recognition and rescoring; corrects it the way a
# person would; then runs the identical page again to show what the system
# learned from that correction.

set -euo pipefail

cd "$(dirname "$0")/.."

DB=${DB:-/tmp/rosetta-demo/rosetta.db}
DATA=${DATA:-/tmp/rosetta-demo/data}
PORT=${PORT:-8810}
RESCORER_PORT=${RESCORER_PORT:-8811}
WORK=$(dirname "$DB")

PREPROCESS=preprocess/build/rosetta-preprocess
SERVICE=service/bin/rosetta

for binary in "$PREPROCESS" "$SERVICE" preprocess/build/mkpage; do
    if [ ! -x "$binary" ]; then
        echo "missing $binary -- run 'make' first" >&2
        exit 1
    fi
done

rm -rf "$WORK"
mkdir -p "$DATA"

cleanup() {
    [ -n "${SERVICE_PID:-}" ] && kill "$SERVICE_PID" 2>/dev/null || true
    [ -n "${RESCORER_PID:-}" ] && kill "$RESCORER_PID" 2>/dev/null || true
}
trap cleanup EXIT

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1. Synthesising a page"
preprocess/build/mkpage -o "$WORK/page.png" \
    --skew 2.5 --perspective 0.08 --shading 0.5 --strike 5 --noise 0.02

say "2. Cleaning it up"
"$PREPROCESS" "$WORK/page.png" -o "$WORK/clean.png" --strikethrough --json - -q \
    | python3 scripts/show_meta.py

# A rescorer left running from an earlier attempt is the worst kind of stale
# state here: this script would quietly talk to a model that had already
# learned from a previous run, and report a flawless first pass that proves
# nothing. Refuse to start rather than produce a result that looks good.
port_in_use() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1 && exec 3>&- && return 0
    return 1
}

for port in "$PORT" "$RESCORER_PORT"; do
    if port_in_use "$port"; then
        echo "port $port is already in use -- stop whatever is on it first" >&2
        echo "(a rescorer left over from an earlier run will make this demo lie)" >&2
        exit 1
    fi
done

say "3. Starting the rescorer, seeded with existing notes"
(cd rescorer && python3 -m rosetta_rescorer --db "$DB" ingest data/sample-notes.txt >/dev/null)
# exec so $! is the python process itself: without it this is the subshell's
# pid, and the cleanup trap kills a shell that has already exited while the
# rescorer keeps the port.
(cd rescorer && exec python3 -m rosetta_rescorer --db "$DB" serve --port "$RESCORER_PORT" \
    >"$WORK/rescorer.log" 2>&1) &
RESCORER_PID=$!

for _ in $(seq 1 40); do
    port_in_use "$RESCORER_PORT" && break
    sleep 0.25
done
if ! port_in_use "$RESCORER_PORT"; then
    echo "the rescorer did not start; see $WORK/rescorer.log" >&2
    exit 1
fi

say "4. Starting the service"
"$SERVICE" -db "$DB" -data "$DATA" -static web/dist -preprocess "$PREPROCESS" \
    -rescorer "http://127.0.0.1:$RESCORER_PORT" -addr "127.0.0.1:$PORT" \
    >"$WORK/service.log" 2>&1 &
SERVICE_PID=$!

for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:$PORT/api/healthz" >/dev/null 2>&1 && break
    sleep 0.25
done

say "5. Uploading the page and running the whole pipeline"
curl -sf -X POST -F "file=@$WORK/page.png" "http://127.0.0.1:$PORT/api/documents" >/dev/null
sleep 4

BASE="http://127.0.0.1:$PORT" RESCORER="http://127.0.0.1:$RESCORER_PORT" \
    DB="$DB" python3 scripts/demo_report.py
