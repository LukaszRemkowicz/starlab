# Follow-up: duplicate-code cleanup

Date: 10 November 2026. This pass addresses remaining clone warnings and the IDE
return-type warning after the initial whole-repository review. Work was performed
in the primary checkout. Existing local formatting changes were preserved.

## Consolidated behavior

- **Record blending:** `catalog.blend_record` owns the common field/RGB updates
  and exact float32 rounding order. `merge_candidate` still counts a fresh source
  as one observation; `combine_records` retains the incoming aggregate count.
  Counts saturate at 255 without silently changing their relative weight policy.
- **Radius decisions:** `numeric.large_size_step` owns the common background
  normalization and adjustment decision, including missing supports, zero
  denominator/NaN comparisons, the 128-pixel growth limit and reversal termination.
  `refinement_result` owns the shared final convergence rule.
- **Ray repair and center:** the Python and Numba paths use the same scalar
  repair and recenter functions from `rays.py`. Numba registers these functions
  as jitable. Backend wrappers still preserve their original input preparation:
  reference inputs are rounded into a fresh list; the compiled wrapper copies
  its float64 array. Marking reads a frozen neighbor snapshot; repairs do not
  cascade through newly replaced neighbors. Both wrappers protect caller data.
- **TIFF drag/drop:** shared acceptance/emission helpers in `drop_paths.py` replace
  the repeated event handling in the empty canvas and image view. First eligible
  local TIFF selection and emission-before-accept order remain unchanged.
- **Source-filter masks:** one helper finalizes masked reduction/enlargement
  buffers with the same bounds, sampling, in-place update and cancellation order.

`candidate()` now constructs an explicit `CandidateRecord`. The accumulated RGB
array has a precise dtype cast at the NumPy boundary. This resolves generic-dict
inference at the return without changing the object layout or calculations.
The type correction is a separate commit from the duplicate refactor.

## Verification

| Check | Result |
|---|---|
| Pylint 4.1.3, `duplicate-code`, similarity setting 5 | 5 reported groups before; 0 after. |
| Gate mutation probe | A copied function in two temporary files was rejected. |
| Full ordinary test suite | 400 passed; 20 explicitly skipped native-GPU cases. |
| Actual OpenGL suite | 20 passed on the desktop GPU. |
| Ruff + formatting | Passed; 90 Python files formatted. |
| Strict mypy | Passed for all 50 production modules. |
| Pre-refactor frozen values | Exact merge fields/counts and repaired centers preserved. |
| Full 80 MP catalog | 31,223 stars; bit-identical position/brightness/RGB/radius records. |
| Full 218 MP panorama catalog | 49,305 stars; bit-identical records in all three checks. |

The fresh-source versus aggregate-count regression deliberately supplies 99
incoming observations at saturation. It verifies that the two different count
policies survived extraction. Radius tests check both reference statistics and
compiled tuple statistics, invalid support, growth reversal and boundary values.
They assert outputs rather than source structure or documentation text.

The detector took 2.676 s on the 80 MP validation run. Panorama checks took 6.216,
4.686 and 5.066 s; the median 5.066 s is close to the previous 5.011 s median. The first
higher run was retained and repeated instead of being treated as proof of a
regression or omitted from the three-sample result. These are prepared detector
measurements, not full application-startup guarantees.

Raw clone reports are saved locally as `.artifacts/review/duplicates-before.json`
and `duplicates-after.json`. Photographic checks are recorded under the existing
ignored benchmark directory with `dedup` labels. Captured regression fixtures
remain unchanged. No clone warning was suppressed or hidden by increasing the
similarity threshold.

## Remaining backend boundary

Array sampling, caching, sparse-kernel support and eight-lane variance evaluation
still have Python/reference and compiled implementations where their data access
is different. CPU/GPU code also remains distinct across execution languages.
Shared scalar policies now have one implementation; forcing different execution
adapters into one generic callback framework would add complexity and needs its
own performance and correctness justification.
