# Review results: items 2–12

Reviewed and implemented on 10 November 2026 in the isolated publication worktree.
The original development checkout was not edited. The approved initial publication
commit remains an ancestor; this work adds commits instead of rewriting it again.

## Changes by item

| Item | Result | Commit |
|---|---|---|
|2 | Validated and documented analysis transfer arrays; returned data are read-only. | `6ceabb1` |
|3 | Shared duplicate ray growth, tile preparation, Qt image conversion, drop parsing and history restoration; removed an unused scanner. Preserved independent reference/JIT backends. | `2cfafea` |
|4 | Used `match` for categorical job/view/effect dispatch. Retained numerical guards and interval checks. | `0859a4e` |
|5 | Validated archive metadata, dimensions, dtypes, compressed-data identity and reconstructed color-cube identity. | `9c0d7d8` |
|6 | Split detection, photometry, rendering and Qt orchestration stages. Fixed import rollback, mixed overview/detail modes, stale labels, worker shutdown and dialog lifetime. Added integer/bounds validation. Enabled complexity gates. | `cfcbbd0` |
|7 | Documented production functions; added array/record/callback/protocol and local state types, strict mypy, annotation/doc rules and Windows CI. Completed private/overload contracts in a documentation follow-up. | `9c69134`, `984df93` |
|8 | Documented the fixed Secondary direction/weight pattern and derived Python counts from its data. Added rotation/unit regressions. | `c0c5016` |
|9 | Documented executable shader stages, pass IDs, coordinate conventions and CPU/GPU differences; named GLSL pass thresholds/capacity without changing arithmetic. | `1f1ec24` |
|10 | Made `ColorTransform` own lazy immutable universal data and its initialization lock. Added concurrent-first-use and failed-load retry tests. Kept pure numerical work as functions. | `92c9acb` |
|11 | Strengthened color, source matching, encoded mask input and composition oracles. Added explicit test backends and real-GPU integration. A new regression exposed a missing JIT dependency in persisted-catalog identity; the fingerprint now includes it. | `6f2b2fb` |
|12 | Confirmed experiment scripts remain excluded from the repository and distributions, with no runtime/test dependency. No production change was necessary. | Documentation and final verification |

The initial review and implementation plan are recorded in `39f3c84`.

## Asset answers

- `analysis_luts.npz` stores fixed transfer arrays: 256 float32 analysis-read levels
  and 4,096 uint8 write levels. It is universal numeric data, not a photo cache.
- `full_lcms_lut.npz` stores a lossless RGB8 transform; its 236,561-byte compressed
  representation reconstructs a 48 MiB cube. The matching JSON describes encoding,
  parameters, library provenance and integrity identities. Both files are needed.
- `SECONDARY_RAYS` is 84 ordered angle/weight pairs. Angles use 0.1 degree units and
  stored weights use percent. It defines appearance, independently of detection.
- `shaders/` contains executable GLSL used by the GPU preview. Vertex code places
  quads; fragment code evaluates rays, halos, rings and source corrections.
- `tools/` contained offline experiments. It is absent from the published tree;
  tests instead use checked-in TIFF/JSON fixtures. See [Assets](assets.md) for detail.

## Verification

| Check | Observed result |
|---|---|
| Ordinary complete suite | 380 passed, 20 explicitly skipped GPU cases; 17.73 s on this machine. |
| Actual native OpenGL suite | 20 passed; 2.99 s on NVIDIA GeForce RTX 3070/PCIe/SSE2. |
| Static lint | All configured Ruff rules passed. |
| Formatting | 89 Python files passed Ruff format checking. |
| Strict type checking | All 50 production modules passed mypy. |
| Production function audit | 438 functions/methods/nested functions; all have docstrings and return annotations. |
| Argument annotations | Two deliberately unannotated overload-factory parameters remain; Numba requires matching annotations between factory and implementation. Other explicit parameters are annotated. |
| Build | Wheel and source distribution built successfully with uv. |
| Package contents | All seven runtime assets present. No experiment directory, virtual environment, cache, Git metadata or generated artifact leaked into either distribution. |
| Package smoke check | Extracted-wheel imports and immutable lookup loading passed; CLI help returned successfully. |
| Publication audit | No tracked experiment scripts or imports; no forbidden legacy identity in tracked paths/content. |

GPU tests exercise all effect groups, gray/RGB, overlapping composition, both source
corrections, halo-independent Soft filter behavior, masked original-resolution
regions, reduced radial bounds/repeatability, all five window view modes, persistent
worker ownership and resource cleanup. Primary/Soft/Ring maximum measured absolute
errors were at most 9.54e-7; Secondary reached 5.44e-4 at a few nearly diagonal pixels.
The test budgets are 2e-5 and 1e-3 respectively, in linear-light units.

The independent pixel oracles cover Screen multiplication, correction clamp/order,
sloped local backgrounds and maximum overlap. Controlled in-memory incorrect
implementations failed the intended assertions for nine cases: ignored saturation,
zero light, reused catalogue references, wrong model-input domain, additive layers,
wrong correction clamp order, omitted gradients and summed shrink/grow overlap.
Captured rendering fixtures remain intact. GUI orchestration uses deterministic
analysis; a separate numerical UI correctness test permits cold JIT compilation.
Mouse-drag tests continue to exercise real Qt gestures before release.

## Remaining boundaries

The complexity limits are 10 for McCabe and 50 statements. Two narrow documented
exceptions remain for the exact rolling accumulator and ordered conservative peak
rejection; preserving their numerical order is intentional. Third-party dynamic
boundaries require a few targeted casts or typed suppressions, particularly Numba
and Qt pointer stubs. No blanket `Any` annotations or module-wide complexity
exclusions were introduced. Static typing does not prove ndarray shape or scientific
accuracy.

This review preserves established algorithms and improves ownership and checks.
It does not establish universal detector recall, physical optical equivalence,
sub-second interaction on every photograph, a 10-second cold start, or support for
all GPU drivers. The actual GPU run covers one Windows/NVIDIA configuration;
macOS runtime and packaging remain unverified. Source fitting in crowded/saturated
fields remains approximate. Further image-quality changes need independent image
measurements, not only green backend comparisons.
