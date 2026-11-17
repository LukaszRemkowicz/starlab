# Packaging and installers

Star Effects Lab is currently installed with `uv sync` and started with
`uv run star-effects-lab`. Native installers for Windows and macOS are planned.
This page records the requirements those installers must meet.

## First launch

The wheel build hook compiles the application's image kernels on synthetic night skies
for a generic CPU of the target architecture. It includes the resulting machine code in
both the standard platform wheel and the editable install made by `uv sync`.
Installers should distribute the wheel for their platform.

On first launch the desktop app extracts the bundled kernels to its writable cache.
Numba uses installation-specific module paths, so the app maps the bundled relative
paths to the installed source directories. A fingerprint covers the numerical source,
Python, Numba, NumPy, LLVM binding version and architecture. Edited sources and custom
CPU targets can still compile locally. Runtime preparation diagnostics for that
fallback go to `logs/preparation.log`.

### App name and icon

Run from source (`uv run star-effects-lab`), macOS lists the app as `python3`
with a blank icon in Force Quit and Activity Monitor, because the process is the
bare Python interpreter. The Dock icon already comes from the code. Only a real
application bundle fixes the rest:

- **macOS:** a `Star Effects Lab.app` bundle whose own executable starts the
  app, with `CFBundleName`/`CFBundleDisplayName` set to `Star Effects Lab` and
  `CFBundleIconFile` pointing to an `.icns` made from
  `presentation/qt/assets/application.png`. A shell-script launcher that runs
  `python3` does not help, because the process would still be `python3`.
- **Windows:** an `.exe` with `presentation/qt/assets/application.ico` embedded,
  so Task Manager and the taskbar show the name and icon.

### Acceptance check for any installer

On a clean machine, install, start the app, open a photo straight away, and read
`startup.log`: the time from `Job started: analyze` to `Worker idle` must be about
a second, not tens of seconds. The log location is listed in the README. Force Quit, Activity Monitor and
Task Manager must show `Star Effects Lab` with its icon.
