#!/usr/bin/env bash
# Measure this machine's per-frame cost on both backends, from the built site.
#
# Usage:
#   scripts/measure_backend.sh [frames] [port]
#
# Serves `docs/` -- the artefact that is actually published -- drives its
# `?measure=1` mode in headless Chrome, and prints one line per execution
# provider. Every line carries the `glRenderer` string, because a figure whose
# renderer is a software rasteriser is not a hardware measurement and must not
# be quoted as one.
#
# This script exists because a figure that cannot be regenerated is a memory
# rather than a measurement. Run it, read the numbers off, and if you quote them
# anywhere quote the renderer with them.
#
# Requires: a built `docs/` (`npm run build` in web/), python3, and Chrome.

set -euo pipefail

FRAMES="${1:-120}"
PORT="${2:-4173}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "$ROOT/docs/index.html" ]]; then
  echo "measure_backend: docs/index.html is missing; run \`npm run build\` in web/ first" >&2
  exit 2
fi

"$PYTHON" -m http.server "$PORT" --bind 127.0.0.1 --directory "$ROOT/docs" > /dev/null 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Give the server a moment, then confirm it is actually answering rather than
# racing the first measurement against a socket that is not listening yet.
for _ in $(seq 1 20); do
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT/index.html"; then
    break
  fi
  sleep 0.25
done

for EP in webgpu wasm; do
  # wasm is single-threaded here (GitHub Pages cannot send COOP/COEP), so it is
  # given fewer frames and a longer deadline: it is roughly five times the cost
  # per frame and the point of the run is the median, not the sample size.
  if [[ "$EP" == "wasm" ]]; then
    COUNT=$(( FRAMES / 2 )); DEADLINE=600
  else
    COUNT="$FRAMES"; DEADLINE=300
  fi
  "$ROOT/scripts/verify_page.sh" \
    "http://127.0.0.1:$PORT/index.html?measure=1&frames=$COUNT&ep=$EP" \
    MEASURE "$DEADLINE"
done
