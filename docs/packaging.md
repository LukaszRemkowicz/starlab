# Packaging and installers

Star Effects Lab is currently installed with `uv sync` and started with
`uv run star-effects-lab`. Native installers for Windows and macOS are planned.
This page records the requirements those installers must meet.

## Requirement: no first-run compilation wait

The detection computation itself takes about 0.03 s on a 512 × 512 photo; with
opening, sky mask and photometry a real detection takes about 0.6 s. The numerical kernels are written with
Numba and are compiled to machine code on first use, which takes about 20 s on a
current Mac. Numba stores the compiled code next to the package (`__pycache__`),
so later launches load it in well under a second.

Measured on macOS (Apple silicon) with the 512 × 512 `pano-sky-*` test fixtures,
before preparation covered every input layout:

| Launch | Window | Star detection | Ready preview |
| --- | --- | --- | --- |
| First after install, nothing compiled | 0.4 s | waits ~21 s once | ~24 s |
| Every later launch | 0.4 s | 0.6 s | ~2.7 s |

A 20 s wait in front of the user is not acceptable. An installer must compile the
kernels before the user first opens a photo.

### How the app behaves today

At every launch `bootstrap.start_preparation` starts a background process
(`starlab.preparation`). It first compares a stamp file (`prepared-code` in the
app's data folder) with a signature of this build: a hash of the package sources,
Python, Numba and CPU type. If they match and Numba's compiled-code cache exists,
it exits at once (about 0.1 s).

Otherwise it runs `application.warmup.warm_up`. That function writes small
synthetic star fields to a temporary folder and runs the app's real open,
thumbnail, sky-mask, detection, preview and export path on them:

- a 512 px image with hundreds of stars, a large saturated star and a border star,
  saved as 16-bit and 8-bit, compressed and uncompressed, RGB and grayscale;
- a 1300 × 1100 image, so the tiled detector path is compiled too.

Numba compiles separate machine code for each input layout (bit depth,
memory-mapped or decoded, single or multiple tiles, at least 256 stars per tile),
so every one of these is needed. Afterwards it writes the stamp. Measured on an
Apple-silicon Mac: about 40 s once, then detection on real 8- and 16-bit photos
takes 0.04-0.7 s.

Opening a photo and detection wait for that process (`ImageServices.ready`), at
most 120 s, instead of compiling a second time. The process stops when the app
exits or crashes (it watches its stdin pipe). Its errors go to
`logs/preparation.log` and a failure is logged in `startup.log`. No photo or star
data is written to disk; only Numba's compiled code and the stamp are.

### Option 1 (recommended): compile during installation

The installer copies the files and then, as its last step, runs the same
warm-up:

```text
<installed python> -P -c "import sys; from starlab.preparation import main; main(sys.argv[1:])" "<app data folder>/prepared-code"
```

The app itself runs `bootstrap.PREPARE` with the stamp path as its argument;
an installer should do the same.

- The user only sees the installation take about 40 s longer; the first launch
  is already fast.
- The code is compiled for the user's own CPU, so it is the fastest and most
  reliable option.
- **Windows:** an `.msi` or `.exe` installer can run this as a post-install
  action. Run it without a console window.
- **macOS:** a plain `.dmg` (drag to Applications) has no install step. Use a
  `.pkg` installer with a `postinstall` script instead.
- The installed package folder must be writable during that step, or
  `NUMBA_CACHE_DIR` must point to a per-user folder that the app also uses at
  run time.

### Option 2: ship precompiled code inside the installer

Run the warm-up while building the installer and package the resulting Numba
cache files.

- No wait even during installation; works with a plain macOS `.dmg`.
- The code must be compiled for a generic CPU (`NUMBA_CPU_NAME=generic`), so it
  runs somewhat slower.
- Fragile: Numba silently recompiles when the source path, file timestamps,
  Python or Numba version, or CPU target differ, and the 20 s wait returns.
  This needs testing on clean Windows and macOS machines before relying on it.

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
