#!/usr/bin/env bash
# Build the Star Effects Lab Photoshop filter on macOS and check it against the Python app.
# Usage: photoshop/build.sh [--sdk DIR] [--app LAUNCHER]
#   --sdk  Adobe Photoshop SDK (the folder holding pluginsdk/, or pluginsdk/ itself). Default: photoshop/sdk/mac
#   --app  star-effects-lab launcher the filter starts. Default: this repo's .venv/bin/star-effects-lab
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
sdk="$here/sdk/mac"
app="$repo/.venv/bin/star-effects-lab"

while [ $# -gt 0 ]; do
    case "$1" in
        --sdk) sdk="${2:?--sdk needs a folder}"; shift 2 ;;
        --app) app="${2:?--app needs a path}"; shift 2 ;;
        -h|--help) sed -n 2,5p "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -d "$sdk/pluginsdk" ] && sdk="$sdk/pluginsdk"
if [ ! -d "$sdk/photoshopapi/photoshop" ]; then
    echo "Photoshop SDK not found at: $sdk" >&2
    echo "Download the Photoshop Plug-In SDK from https://developer.adobe.com/console (free, Adobe login)," >&2
    echo "unzip it to photoshop/sdk/mac (so photoshop/sdk/mac/pluginsdk exists), or pass --sdk DIR." >&2
    exit 1
fi
for tool in cmake Rez codesign; do
    command -v "$tool" >/dev/null || { echo "Missing $tool (install Xcode Command Line Tools and cmake)." >&2; exit 1; }
done
[ -x "$app" ] || echo "Warning: launcher not found at $app; the filter will fail until it exists (uv sync)." >&2

cmake -S "$here/plugin" -B "$here/build" -DPHOTOSHOP_SDK="$sdk" -DSTARLAB_COMMAND="$app"
cmake --build "$here/build" --parallel

python="$repo/.venv/bin/python"
if [ -x "$python" ]; then
    "$here/build/bridge_selftest" "$python"
else
    echo "Skipping bridge self-test: $python not found (run uv sync)." >&2
fi

echo
echo "Plug-in: $here/build/Star Effects Lab/Star Effects Lab.plugin"
echo "Install: link or copy the folder '$here/build/Star Effects Lab' into"
echo "         /Library/Application Support/Adobe/Plug-Ins/CC/ and restart Photoshop."
