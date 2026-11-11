# Code review: items 2–12

Reviewed baseline: `2780097`, 10 November 2026. Scope: all 44 production Python modules, both GLSL shaders, 28 Python test modules, packaged numerical/model data, build configuration and fixture manifests. Review preceded implementation. Independent read-only passes covered detection, Qt, and effects/tests; the coordinating pass covered application, I/O, assets and cross-layer behavior.

Implemented changes, final checks and commit references are recorded in [Review results](review-results.md). Findings below describe the reviewed baseline, including defects subsequently fixed.

## Baseline evidence

- Full suite: 341 tests passed in 16.79 s.
- 328 functions/nested functions/methods; 287 have no function docstring, 230 no return annotation, 214 at least one untyped explicit argument.
- Existing Ruff selects only `E4,E7,E9,F,I`. A separate complexity run reports 21 `C901` findings and 88 findings across selected complexity/branch/statement/argument rules. These findings are diagnostic, not proof of defects.
- The largest method is GPU preview `__call__` at 283 lines; detection orchestrators reach 146–154 lines; Qt completion handling is 104 lines.
- No tests assert documentation prose. Numerical fixtures and compiled/reference comparisons are useful, but they do not establish production GPU correctness.

## Explanations requested

| Item | Actual role and decision |
|---|---|
|2: `analysis_luts.npz` | NumPy archive, 5,624 bytes. `read`: 256 float32 entries, analysis-domain decoding; `write`: 4,096 uint8 entries, quantized encoding after normalization. It is universal transfer data, not a photograph or saved detection result. Stored read values differ slightly from freshly evaluating a gamma power, so casually regenerating them changes numerical behavior. Keep, document, validate and make arrays immutable. |
|5: `full_lcms_lut.json/.npz` | Universal RGB8 color transform. The 236,561-byte archive stores a uint8 delta cube of shape 256×256×256×3 plus three 256-entry channel baselines; reconstructed data occupy about 48 MiB. JSON records encoding, transform parameters, versions and hashes. Runtime currently checks the reconstructed table hash; archive and adapter hashes are provenance, not enforced checks. Keep both data and metadata; document that distinction and validate loading. |
|8: `SECONDARY_RAYS` | 84 fixed direction/weight pairs, not an abstraction layer. Angles use tenths of a degree; weights use integer percent. The irregular pattern defines secondary-flare appearance. There is only one production table. Keep its values/order, make units and derived count explicit. Detection's eight-direction sampling stencil serves a different purpose. |
|9: `shaders/` | Executable GLSL for the GPU preview. The vertex shader places source/ray quads; the fragment shader evaluates profiles, antialiasing and source corrections. GPU blending contributes Screen/MAX operations; final image composition is outside these shader functions. Python/GLSL mathematical duplication is unavoidable across execution backends and needs real GPU parity tests. |
|12: `tools/` | Former local measurement, comparison and experiment scripts. They are excluded from the published tree and are not runtime dependencies. Rendering fixtures were made self-contained before that exclusion. Production diagnostics or tests must not import a local experiment directory. |

## Ranked correctness and coverage findings

### P1: progressive import can discard the current document

`presentation/qt/controller.py:185` replaces Session/source/previews as soon as a thumbnail arrives, before full import succeeds. Cancellation/failure cannot restore the edited previous document. A focused offscreen probe reproduced an empty committed session and lingering progressive-open state. Keep document state committed until full import succeeds; use temporary presentation state and restore the previous display on failure/cancellation.

### P2: zoomed mode changes mix old overview with new detail

`controller.py:385` updates only the region item when a detail preview is present. The underlying overview can remain Result while the visible region becomes Original/Mask/Effects. A direct probe showed different overview/detail content in Original mode. The disabled-effects update also rebuilds the overview as Result regardless of mode. Centralize mode-aware pixel conversion and refresh both layers.

### P2: shutdown timeout leaves an open but disabled worker

`presentation/qt/worker.py:90` permanently sets `closing=True` before returning a failed shutdown. MainWindow then stays open, but subsequent submissions silently return. Make incomplete draining recoverable, preserve single-thread GPU ownership, and release resources exactly once.

### P2: production GPU/sky wiring is not exercised by the ordinary suite

`tests/conftest.py` selects offscreen Qt; bootstrap omits GPU and real mask services in that mode. The image regressions cover CPU. Add an explicit real-OpenGL suite for shader compilation, full-resolution CPU/GPU tolerance, gray/RGB, masks, ROI, both corrections and cleanup. Reduced radial integration must use appropriate invariants rather than equality with CPU point samples.

### P2/P3: weak test oracles

- `test_sky_model.py:65` claims to verify encoded TIFF input, but the fake network never examines received tiles.
- `test_effect_color_defaults.py:25` can pass if nonzero saturation is ignored or no effect is emitted.
- Catalogue comparisons independently choose nearest references, allowing duplicate detections to reuse one expected source.
- All 42 rendering fixtures activate one effect group and contain no source-correction profiles. Keep them, and add small independent algebraic oracles for combined composition/correction pixels.
- Some UI tests use sleeps or tiny deadlines. Separate functional state tests from compilation/performance measurements and wait for events/conditions.

### P3: smaller state/lifecycle problems

Cancellation/failure can leave the Detecting stars label; star-edit dialogs remain parent-owned after closing. The unused `candidate.detect_candidates` helper uses incorrect coordinates for a nonzero image origin; production does not call it. Remove the unused scanner rather than maintain another scanning path. Domain mask/detection parameter validation also needs explicit integer/bounds checks rather than implicit truncation or downstream failures.

## Maintainability findings

### Item3: real duplication versus intentional backend equivalence

Real shared work: ray coefficient setup in profiles/GPU; duplicate ray sample-growth loops inside compiled initial tracing; repeated tile-bounds/mask preparation; duplicated QImage conversion; duplicated undo/redo restoration; repeated source-correction masking. Extract shared stages without changing arithmetic or ownership.

Separate Python/Numba candidate, ray, join and sparse-filter implementations intentionally provide a readable reference and optimized typed buffers. Preserve independent reference oracles; a callback/class hierarchy across JIT kernels would be a regression. Floating-point helpers impose explicit float32 rounding; they are not redundant wrappers around ordinary float64 arithmetic.

### Item4: use categorical dispatch, preserve numerical guards

`match` fits job-result kinds and display modes. Dictionaries already fit simple label/gain lookup. Interval thresholds, early rejection, clipping, mask checks and numerical boundary cases should remain `if`; changing syntax alone does not reduce responsibility or mathematical complexity. Match inside Numba kernels is not assumed to compile.

### Item6: responsibilities and quality gates

Extract GPU crop/catalogue/target/group/correction/readback stages; detector tile preparation/scheduling/source qualification; multiscale per-tile seed generation; mask preprocessing/inference; and Qt completion/display stages. Preserve exact rolling-filter accumulator order and other numerical state-machine invariants. Enable meaningful complexity gates, with narrow documented exceptions only where mathematical state-machine structure justifies them.

Ruff has a [McCabe complexity rule](https://docs.astral.sh/ruff/rules/complex-structure/), but it was not enabled. Its passing baseline did not establish low complexity, duplication freedom, docstring coverage or complete typing.

### Item7: documentation and actual types

Add responsibility docs, public function contracts, shape/dtype/coordinate descriptions and private numerical invariants. Add real array/callback/record types at boundaries, meaningful local annotations for heterogeneous mappings and asynchronous payloads, and a type-checking command. Do not annotate every obvious loop index or replace unknown types with blanket `Any` merely to silence checks. Ruff's [public-function documentation rule](https://docs.astral.sh/ruff/rules/undocumented-public-function/) and a [type checker](https://mypy.readthedocs.io/en/stable/config_file.html) address different gaps.

### Item10: classes by ownership

Existing Image, Session, ArrayCache, mask services, Qt widgets and GPU renderer justify objects because they own state/resources. Frozen settings/source records justify value objects. Extract smaller owners where state is mixed; keep numerical kernels as functions. More classes alone are not a correctness or maintainability metric. No detector inheritance tree or one-class-per-effect framework is justified.

### Item11: test scope

Keep captured-image/catalogue fixtures, recorded ONNX logits, analytical photometry, compiled/reference differential tests, source-correction invariants, and genuine UI behavior checks. Strengthen weak oracles and backend coverage. Avoid tests that inspect implementation structure or documentation wording. Separate backend equivalence, image-quality expectations, UI workflows and timing measurements.

## Preservation constraints

No fastmath, algebraic reassociation, float dtype changes, generic sum replacements or changed operation order without numerical evidence. Preserve float32 rounding sites, eight-lane variance accumulation, merge traversal/count semantics, direction/tie order, quarter-pixel conventions, global normalization, exact encode16-before-shift behavior and rolling-filter state. Keep Qt image-copy ownership, GUI-thread pixmaps, persistent single GPU job thread, cancellation/revision semantics and original-pixel immutability.

## Implementation record

Each numbered item that changes production or test code receives its own commit. Documentation-only explanations can share this initial review record. Baseline measurements above remain historical; final checks and commit references will be recorded separately as the work proceeds.
