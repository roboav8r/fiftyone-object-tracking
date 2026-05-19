#!/bin/bash
# Symlink this plugin into the FiftyOne plugins directory for local
# development. The plugin loader picks it up by directory name; the JS
# bundle is loaded from dist/index.umd.js.
#
# Usage:
#   ./install.sh                              # symlink into ~/fiftyone/__plugins__
#   FIFTYONE_PLUGINS_DIR=/path ./install.sh   # custom plugin root
set -euo pipefail

PLUGIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLUGIN_NAME="$(basename "$PLUGIN_DIR")"
ROOT="${FIFTYONE_PLUGINS_DIR:-$HOME/fiftyone/__plugins__}"

mkdir -p "$ROOT"
TARGET="$ROOT/$PLUGIN_NAME"

if [[ -L "$TARGET" || -e "$TARGET" ]]; then
    echo "Already installed at $TARGET — removing first." >&2
    rm -rf "$TARGET"
fi

ln -s "$PLUGIN_DIR" "$TARGET"
echo "Symlinked: $PLUGIN_DIR -> $TARGET"
echo
echo "Restart the FiftyOne App (or 'fiftyone app debug') to pick up the plugin."
echo "Verify with:  fiftyone plugins info @roboav8r/fiftyone-object-tracking"
