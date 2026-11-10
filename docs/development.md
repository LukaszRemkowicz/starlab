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
uv run mypy
uv run pylint src/starlab
```

Ordinary Qt tests explicitly request the CPU test backend and use the offscreen platform. They exercise application workflows without driving a user's desktop; they do not silently stand in for production GPU/model wiring. Numerical tests cover image validation, color transforms, detector regressions, CPU kernel agreement, masks, source corrections, rendering profiles, history and stale preview handling. Some UI tests have timeouts and require generic Numba specializations to be available; a first run after numerical-code edits can compile them.

Ruff checks selected syntax/import/error rules, import ordering, unused suppressions, McCabe complexity (maximum 10) and statement count (maximum 50). Two function-scoped exceptions document numerical state machines whose exact accumulator/rejection order is intentional: the two-axis rolling filter and conservative sparse peak screening. No module-wide complexity suppressions are used. Annotation and public-documentation rules are also enabled. Mypy checks all production modules in strict mode; array aliases distinguish dtype, and protocols describe service boundaries. The signature-preserving JIT adapter keeps static checking across Numba decorators without changing runtime compilation. Two overload-factory parameters remain intentionally unannotated because Numba requires their annotations to equal the implementation parameters. Untyped external-library stubs are isolated by boundary casts; strict checking does not prove array shape or numerical accuracy. Independent image and workflow checks remain necessary.

The Windows quality workflow runs the locked environment, lint, formatting, strict type checking, ordinary tests and wheel build on pushes and pull requests. It follows the [official uv Actions integration](https://docs.astral.sh/uv/guides/integration/github/). A hosted headless test run does not establish hardware shader correctness.

## Duplicate-code checks

`uv run pylint src/starlab` enables only Pylint's
[`duplicate-code` rule](https://pylint.pycqa.org/en/latest/user_guide/messages/refactor/duplicate-code.html).
The similarity setting is 5 meaningful lines; comments, docstrings, imports and
signatures are ignored. CI runs this separately from Ruff and mypy. No production
module is excluded and no duplicate-code suppression is used. A controlled copied
function was rejected to verify that the gate actually runs.

Pylint compares similar text across modules; PyCharm can also identify renamed
structural copies and same-file duplication. A passing clone scan is one check,
not proof that every conceivable similar algorithm has been eliminated. Review
must still distinguish shared business rules from backend-specific data access.
See [the duplicate cleanup](duplicate-cleanup.md) for the measured scope.

## Native GPU checks

Run the opt-in suite on a desktop with a working OpenGL 3.3 context:

```powershell
uv run pytest tests/test_gpu_integration.py --run-gpu -q
```

The test configuration selects the native Qt platform for this command. Missing
context support is a failure when explicitly requested; the ordinary suite reports
these cases as skipped. The suite exercises actual shader compilation, gray/RGB,
all four groups, combined composition, both source corrections, masked regions,
reduced radial invariants, window display modes and worker-owned cleanup.
Primary/Soft/Ring comparisons use a 2e-5 linear-light absolute budget; Secondary
uses 1e-3 because its near-diagonal rasterization differs slightly between CPU
and GPU. These are backend comparison budgets, not camera calibration claims.

Test oracles also include fixed captured images/catalogues, literal high-precision
pixel values for overlap/correction math, and one-to-one catalogue assignment.
Assertions must establish a positive effect before comparing control behavior.
The encoded-mask test examines actual network input instead of merely accepting
constant fake logits. Tests do not assert documentation wording or source layout.

## Layout

- `src/starlab/`: production code and packaged assets.
- `tests/`: automated tests.
- `tests/fixtures/`: self-contained image and numerical regression data.
- `docs/`: maintained English documentation.

Local experiments under `tools/`, environments, IDE configuration and generated artifacts are ignored. Tests and application imports must not depend on those directories. Historical investigation material is preserved outside the published checkout rather than presented as current product documentation.

## Verification scope

Pixel regressions compare numerical output with captured expected images. They establish behavior for those fixtures, not a universal quality claim for every photograph. A full GPU/driver matrix, monitor-profile color management and macOS packaging have not been verified. Startup and interaction timings depend on hardware, source content and compilation state; published documentation makes no universal latency guarantee.
