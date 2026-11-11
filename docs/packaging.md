# Packaging and installers

Star Effects Lab is currently installed with `uv sync` and started with
`uv run star-effects-lab`. Native installers for Windows and macOS are planned.
This page records the requirements those installers must meet.

## Requirement: no first-run compilation wait

Star detection itself takes about 0.03 s. The numerical kernels are written with
Numba and are compiled to machine code on first use, which takes about 20 s on a
current Mac. Numba stores the compiled code next to the package (`__pycache__`),
so later launches load it in well under a second.

Measured on macOS (Apple silicon), 512 × 512 test image:

| Launch | Window | Star detection | Ready preview |
| --- | --- | --- | --- |
| First after install, nothing compiled | 0.4 s | waits ~21 s once | ~24 s |
| Every later launch | 0.4 s | 0.6 s | ~2.7 s |

A 20 s wait in front of the user is not acceptable. An installer must compile the
kernels before the user first opens a photo.

### How the app behaves today

At every launch `bootstrap.start_preparation` starts a background process that
runs `application.warmup.warm_up`. That function runs the real open, sky-mask,
detection (both methods), preview and export path on a small synthetic image in
a temporary folder, so Numba compiles exactly the specializations the app uses.
If the code is already compiled, the process only loads it and exits. Opening a
photo and detection wait for that process (`ImageServices.ready`) instead of
compiling a second time. No photo or star data is written to disk; only Numba's
compiled-code cache is.

### Option 1 (recommended): compile during installation

The installer copies the files and then, as its last step, runs the same
warm-up:

```text
<installed python> -c "from starlab.application.warmup import warm_up; from starlab.bootstrap import create_services; warm_up(create_services(sky_mask=True))"
```

The value of `bootstrap.PREPARE` is this exact command string.

- The user only sees the installation take about 20 s longer; the first launch
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

### Acceptance check for any installer

On a clean machine, install, start the app, open a photo straight away, and read
`startup.log`: the time from `Job started: analyze` to `Worker idle` must be about
a second, not tens of seconds. The log location is listed in the README.
