#!/usr/bin/env bash
# Run a page in headless Chrome and read its verdict out of the tab title.
#
# Usage:
#   scripts/verify_page.sh <url> [expected-prefix] [timeout-seconds]
#
# Exit status is the point: 0 when the title starts with the expected prefix
# (PASS by default), non-zero on the opposite verdict, on a timeout, or when
# Chrome never came up. That makes it usable from CI, from a release check, and
# from a mutation test that needs to watch this go red.
#
# Two traps this script exists to avoid, both measured on macOS:
#
#   * `--virtual-time-budget` freezes the media stack. A page that waits on a
#     <video> or on getUserMedia never settles under it, so it is not used here
#     however tempting a deterministic clock is.
#   * There is no supported way to read a headless tab's title from the command
#     line other than the DevTools endpoint. So the page writes its verdict into
#     document.title and this script polls http://127.0.0.1:<port>/json, which
#     needs no code injected into the page and no CDP session.
#   * That endpoint lists more than the tab: a fresh headless Chrome also
#     reports an "Omnibox Popup" target, whose title is not a verdict and which
#     a naive "first title in the JSON" read would return forever. Targets are
#     therefore filtered to type "page" on the URL that was asked for.

set -euo pipefail

URL="${1:-}"
EXPECT="${2:-PASS}"
TIMEOUT="${3:-120}"

if [[ -z "$URL" ]]; then
  echo "usage: $0 <url> [expected-prefix] [timeout-seconds]" >&2
  exit 2
fi

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
if [[ ! -x "$CHROME" ]]; then
  CHROME="$(command -v google-chrome || command -v chromium || true)"
fi
if [[ -z "$CHROME" || ! -x "$CHROME" ]]; then
  echo "verify_page: no Chrome executable found; set CHROME=<path>" >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" > /dev/null; then
  echo "verify_page: $PYTHON not found; set PYTHON=<path>" >&2
  exit 2
fi

PORT="${DEVTOOLS_PORT:-9222}"
PROFILE="$(mktemp -d)"
LOG="$PROFILE/chrome.log"

cleanup() {
  if [[ -n "${CHROME_PID:-}" ]]; then
    kill "$CHROME_PID" 2>/dev/null || true
    wait "$CHROME_PID" 2>/dev/null || true
  fi
  rm -rf "$PROFILE"
}
trap cleanup EXIT

# --headless=new can reach the real GPU, which is the whole point on a page that
# reports which backend it got. Everything else here is about making a
# throwaway profile behave like a fresh install.
"$CHROME" \
  --headless=new \
  --remote-debugging-port="$PORT" \
  --user-data-dir="$PROFILE" \
  --no-first-run \
  --no-default-browser-check \
  --disable-features=Translate,MediaRouter \
  --autoplay-policy=no-user-gesture-required \
  --use-fake-ui-for-media-capture \
  --window-size=1440,1000 \
  "$URL" > "$LOG" 2>&1 &
CHROME_PID=$!

deadline=$(( $(date +%s) + TIMEOUT ))
title=""
while (( $(date +%s) < deadline )); do
  if ! kill -0 "$CHROME_PID" 2>/dev/null; then
    echo "verify_page: Chrome exited early" >&2
    tail -5 "$LOG" >&2 || true
    exit 3
  fi
  title="$(curl -s --max-time 5 "http://127.0.0.1:$PORT/json" | "$PYTHON" -c '
import json, sys
try:
    targets = json.load(sys.stdin)
except Exception:
    sys.exit(0)
wanted = sys.argv[1]
for target in targets:
    if target.get("type") != "page":
        continue
    if target.get("url", "").split("#")[0] != wanted.split("#")[0]:
        continue
    print(target.get("title", ""))
    break
' "$URL" || true)"
  case "$title" in
    "$EXPECT"*)
      echo "$title"
      exit 0
      ;;
    FAIL*|MEASURE-FAIL*)
      echo "$title" >&2
      exit 1
      ;;
  esac
  sleep 1
done

echo "verify_page: timed out after ${TIMEOUT}s; last title: ${title:-<none>}" >&2
exit 4
