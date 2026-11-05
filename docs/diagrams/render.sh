#!/bin/sh
# Renders architecture.html to ../architecture.png (3200 x 1780, a 2x screenshot) with headless Chrome.
# Usage: docs/diagrams/render.sh   (set CHROME to the browser binary if it is not the macOS default)
set -e
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
HERE="$(cd "$(dirname "$0")" && pwd)"
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --window-size=1600,890 --screenshot="$HERE/../architecture.png" "file://$HERE/architecture.html"
