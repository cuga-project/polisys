#!/bin/sh
# Renders architecture.svg to ../architecture.png (3200 x 1860, a 2x image) with headless Chrome. The README embeds the SVG itself.
# Usage: docs/diagrams/render.sh   (set CHROME to the browser binary if it is not the macOS default)
set -e
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
HERE="$(cd "$(dirname "$0")" && pwd)"
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --window-size=1600,930 --screenshot="$HERE/../architecture.png" "file://$HERE/architecture.svg"
