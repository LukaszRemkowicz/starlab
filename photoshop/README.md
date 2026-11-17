# Star Effects Lab Photoshop filter

A small C++ filter plug-in (Filter > Star Effects Lab > Star Effects Lab...). It copies the active layer
(RGB or Grayscale, 8 or 16 bit) into a shared-memory block, runs `star-effects-lab --photoshop-bridge NAME`,
and copies the result back when you press Apply. Pixels never touch the disk. The block layout is defined by
`src/starlab/infrastructure/photoshop_bridge.py`.

Protocol v2 (little-endian): 64-byte header `SELB`, version (2), width, height, channels, depth, white code,
status (offset 28), mode (offset 32: 0 window, 1 repeat), settings length (offset 36); then the pixels; then a
fixed 65536-byte settings area holding UTF-8 JSON (detection + effect settings). After a successful Apply the
plug-in keeps that JSON in the filter's `parameters` handle; Filter > Last Filter (Cmd/Ctrl+F) then starts the app
in repeat mode, which re-applies the stored settings with no window.

## Build

The Adobe Photoshop SDK is not in this repository (its licence forbids redistribution). Download the
Photoshop Plug-In SDK once from https://developer.adobe.com/console (free, Adobe login) and unzip it to
`photoshop/sdk/mac` (macOS) or `photoshop/sdk/win` (Windows), or pass its folder to the script.

    photoshop/build.sh [--sdk DIR] [--app LAUNCHER]          # macOS: cmake + Command Line Tools
    photoshop\build.ps1 [-Sdk DIR] [-App LAUNCHER]           # Windows: cmake + VS Build Tools (C++), x64

The script builds the plug-in and runs `bridge_selftest` (protocol check against the Python app, no Photoshop).
`--app` defaults to this repo's `.venv` launcher.

Result: `photoshop/build/Star Effects Lab/Star Effects Lab.plugin` on macOS (universal arm64 + x86_64,
ad-hoc signed); `photoshop\build\Release\StarEffectsLab.8bf` on Windows (untested so far).

## Install

Copy the bundle into `/Library/Application Support/Adobe/Plug-Ins/CC/` or the `Plug-ins` folder of your
Photoshop install, then restart Photoshop.

## How the plug-in finds the app

The launcher path is compiled in as `STARLAB_COMMAND` (CMake cache variable). The default is the developer's
`.venv/bin/star-effects-lab`; an installer is expected to set the real path.
