# Development and verification

## Environment

```powershell
uv sync
uv run star-effects-lab
uv run python -m starlab
```

The distribution and console command are `star-effects-lab`; the Python package is `starlab`. Runtime assets are part of that package. Development dependencies are declared in `pyproject.toml` and resolved in `uv.lock`.

## Checks

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Qt tests use the offscreen platform through their test configuration. They exercise application workflows without driving a user's desktop. Numerical tests cover image validation, color transforms, detector regressions, CPU kernel agreement, masks, source corrections, rendering profiles, history and stale preview handling. Some UI tests have timeouts and require generic Numba specializations to be available; a first run after numerical-code edits can compile them.

The current Ruff configuration checks selected syntax/import/error rules and import ordering. It does not currently enforce complete type coverage, docstring coverage, duplication limits or comprehensive complexity limits. A successful Ruff run does not establish those properties. Type-checker and complexity-policy changes belong to the later code-review work rather than this documentation/identity publication.

## Layout

- `src/starlab/`: production code and packaged assets.
- `tests/`: automated tests.
- `tests/fixtures/`: self-contained image and numerical regression data.
- `docs/`: maintained English documentation.

Local experiments under `tools/`, environments, IDE configuration and generated artifacts are ignored. Tests and application imports must not depend on those directories. Historical investigation material is preserved outside the published checkout rather than presented as current product documentation.

## Verification scope

Pixel regressions compare numerical output with captured expected images. They establish behavior for those fixtures, not a universal quality claim for every photograph. A full GPU/driver matrix, monitor-profile color management and macOS packaging have not been verified. Startup and interaction timings depend on hardware, source content and compilation state; published documentation makes no universal latency guarantee.
