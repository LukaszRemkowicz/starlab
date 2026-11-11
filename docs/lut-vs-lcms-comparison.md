# Universal RGB transform: LUT versus direct LittleCMS

Measured on 10 November 2026 against commit `c7cc8f3`.

## Decision

Keep `full_lcms_lut.npz` and its JSON metadata in the current optimized detector.
The table reduces prepared detector time by **15.7% on the 80 MP image** and
**19.9% on the 218 MP panorama**, with identical catalogs. Its fixed array costs
48 MiB; median native working-set growth during initialization was 48.29 MiB.
Loading and validating it took 53.03 ms, versus 5.36 ms to build a direct transform.
The additional initialization cost is about 48 ms on this machine.

This supports the current optimization with measurements. It does not establish
that a large cube is essential to every detector or every possible implementation.
No production algorithm or asset was changed for this comparison.

## Compared paths

- **LUT:** production `proxy.full_proxy(pixels, denominator)`. A Numba loop fuses
  integer encoding, lookup in the complete RGB cube, and analysis-level decoding.
- **Direct LCMS:** existing reference expression
  `proxy.decode(proxy.full_codes(pixels), denominator)`. It uses the same integer
  encoding and actual Pillow/LittleCMS transform, followed by the same decoding.

For whole-image runs, only the process-local `pipeline.full_proxy` callable was
replaced. `COMPILED` remained true: source qualification, seeding, joining,
threading and all detector parameters were unchanged. Setting `COMPILED=false`
would also change other detector code and would be an invalid comparison here.

The measured benefit belongs to these two complete proxy implementations.
It includes fused passes and reduced intermediate allocations as well as the
cube lookup; it must not be presented as the isolated cost of an ICC function.

## Numerical equivalence

All **16,777,216 RGB8 triplets** were transformed through the installed LCMS and
compared with the packaged cube. Different triplets: **0**. Maximum channel error:
**0**. Direct-transform SHA-256 matched the packaged table:
`9cf814a133376c04853756c212d9524c701e0236ed6e79a8123dfe53a21b5a9e`.

The full proxy results were also bit-identical on the measured RGB/grayscale
arrays. Both whole-image catalogs matched exactly in positions, brightness,
RGB and source radii, including order:

| Source | Dimensions | Stars in either path | Catalog SHA-256 |
|---|---|---:|---|
| `D_pelne_bez_plamy.tif` | 8345 × 9599 | 31,223 | `47be124c9df426f946bbec2f1397dc040897d83d12b099acc9930421360dd583` |
| `pano.tif` | 17595 × 12373 | 49,305 | `722a31d46bc41ffabf56591997a02bc5065af90d0dc59c1f1900613dd31200f7` |

These results apply to Pillow 12.3.0 / LittleCMS 2.19. Future library versions
must be checked again before assuming identical direct conversion.

## Prepared proxy timings

Median of seven calls per path, with alternating order on identical read-only,
non-contiguous input views. Both paths were initialized first. Input generation
and equality checks were outside the timers; the normalization denominator was 0.4.

| Input | LUT | Direct LCMS | Direct / LUT time |
|---|---:|---:|---:|
| 512 × 512 RGB | 10.25 ms | 21.83 ms | 2.13× |
| 1024 × 1024 RGB | 54.68 ms | 89.71 ms | 1.64× |
| 1568 × 1568 RGB | 105.23 ms | 211.18 ms | 2.01× |
| 3072 × 3072 RGB | 328.59 ms | 819.72 ms | 2.49× |
| 1568 × 1568 grayscale | 25.85 ms | 155.39 ms | 6.01× |

The 1568-pixel case represents a 1024-pixel detector tile with 272-pixel halos
on both sides. Synthetic samples cover a broad color range; the real-image
measurements below establish the practical result on photographic inputs.

## Whole-image detector timings

Three separate processes per source/path after preparing generic Numba kernels.
Source TIFFs were read without modification, and a real ONNX sky mask was built
on every run with persistent mask caching disabled. No saved photo catalog was
loaded. Variants ran sequentially in interleaved order.

| Source | LUT detector | Direct LCMS detector | Time saved | Detector-time reduction |
|---|---:|---:|---:|---:|
| 80.10 MP image | 2.821 s | 3.347 s | 0.526 s | 15.7% |
| 217.70 MP panorama | 5.011 s | 6.259 s | 1.248 s | 19.9% |

Recorded detector samples in seconds:

- 80 MP LUT: 2.82149, 2.75000, 2.83895; LCMS: 3.34753, 3.34743, 3.26096.
- 218 MP LUT: 5.17674, 5.01136, 4.80940; LCMS: 6.27293, 6.25932, 6.24435.

These timers include proxy preparation and detection, excluding TIFF reading,
sky-mask construction, subsequent photometry, rendering and GUI presentation.
Median full-resolution detector-phase times were 2.199/2.732 s on the 80 MP image
and 3.738/4.977 s on the panorama, respectively.

An initial calibration run, retained as `image-lut-D-1.json`, took 10.046 s for
the detector and 37.350 s including preparation, reading and mask construction.
It is listed separately from the three prepared runs (`D-2` through `D-4`).
The prepared results above are not cold-JIT, cold-disk or complete application
startup measurements and do not establish a universal 10-second startup limit.

## Memory and initialization

Initialization was measured in three fresh processes per path, before creating
photographic working arrays. Windows native process counters were used.

| Measure, median | LUT | Direct LCMS |
|---|---:|---:|
| Data/transform initialization | 53.03 ms | 5.36 ms |
| Added resident working set during initialization | 48.29 MiB | 0.38 MiB |
| Fixed cube payload | 48 MiB | None |
| Peak whole-process working set, 80 MP run | 2407.38 MiB | 2342.92 MiB |
| Peak whole-process working set, panorama run | 4380.71 MiB | 4333.58 MiB |

Whole-process peaks include source pixels, model work, native runtimes and
concurrent detector buffers. They are not the size of the LUT alone and may
vary with allocation/thread scheduling. The measured panorama peak difference
was 47.13 MiB. Resident working set and private commit are distinct counters;
both are preserved in the raw output.

## Environment and reproduction

Windows x64, 16 logical CPUs; Python 3.14.5, NumPy 2.5.4, Numba 0.68.1,
Pillow 12.3.0 and LittleCMS 2.19. Production uses 16 detector workers,
1024-pixel tiles and 272-pixel halos.

The ignored local directory `.artifacts/benchmarks/lut-vs-lcms-20261110/` contains
`benchmark_lut_lcms.py`, per-run JSON, the exhaustive result and `summary.json`.
The summary records source commit and script SHA-256. The script imports the
normal installed package; it has no dependency on an excluded experiment folder.

From the main checkout, examples using a fresh output label:

```powershell
uv run python .artifacts/benchmarks/lut-vs-lcms-20261110/benchmark_lut_lcms.py exhaustive
uv run python .artifacts/benchmarks/lut-vs-lcms-20261110/benchmark_lut_lcms.py micro
uv run python .artifacts/benchmarks/lut-vs-lcms-20261110/benchmark_lut_lcms.py image --backend lut --source 'D:\photos\image.tif' --label=-new-1
uv run python .artifacts/benchmarks/lut-vs-lcms-20261110/benchmark_lut_lcms.py image --backend lcms --source 'D:\photos\image.tif' --label=-new-1
```

The timing script and raw photographic measurement output stay local. This report
records the measured conclusions; no application source change is required.
