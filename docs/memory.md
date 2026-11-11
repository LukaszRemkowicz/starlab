# Memory and chunked processing

Large panoramas must not need several times their own size in RAM. This page
records where memory goes and how the app processes images in chunks.

## Measured baseline (before chunking)

48-megapixel RGB photo (8000 × 6000, 16-bit), all effects enabled, macOS. The
float32 linear working image alone is 576 MB (12 bytes per pixel). "Peak" is the
resident memory above the level before each stage.

| Stage | Uncompressed TIFF | Compressed TIFF (zlib) |
| --- | --- | --- |
| Thumbnail | +73 MB | +822 MB (whole file decoded) |
| Read working image | +759 MB | +505 MB |
| Sky mask from the file | +879 MB | decoded again in RAM |
| Detection + photometry | +1026 MB | +1160 MB |
| Export | +1769 MB, 243 s | +1915 MB, 262 s |
| Whole session peak | ~2.5 GB (~50 B/px) | ~4.2 GB (~85 B/px) |

For uncompressed files part of the resident size is the memory-mapped source,
which the operating system can drop at any time.

## Design

The original TIFF on disk and one float32 linear working image in RAM stay the
source of truth. Every algorithm (photometry, Soft filter, preview at any zoom)
needs random access to linear pixels, so replacing the working image is out of
scope for now. Everything else is processed in bands or tiles:

1. **Streaming decode** (`infrastructure/tiff_io.EncodedRows`). Encoded samples
   are read band by band: memory-mapped when the file allows it, otherwise only
   the strips or tiles that intersect the requested rows are decoded
   (`TiffPage.decode` on a shared thread pool) and released after use. Reading
   the working image, the thumbnail and the sky mask never hold a whole decoded
   file, except for a compressed TIFF stored as one single strip, which must be
   decoded whole. Nothing is written to disk.
2. **Bounded sky mask** (`domain/sky_mask.py`). Normalization statistics come
   from the reduced image; each model tile is normalized on its own, and
   overlapping logits are kept in a rolling window of rows that are classified
   and released as soon as no later tile touches them.
3. **Bounded detection** (`domain/detection`). One tile in flight per worker;
   tile buffers live on their own anonymous memory mappings
   (`buffers.anonymous_array`), so their pages go back to the system as soon as
   a tile is done instead of staying in the allocator's cache.
4. **Streaming export** (`application/use_cases._export_bands`). Effects are
   rendered for one 256-row band at a time, composited with that band of the
   working image, corrected by the Soft filter, encoded to 16-bit and streamed
   row by row into the TIFF writer. No full-size effect layer, float result or
   16-bit result exists. Ray and radial profiles are cached once and sliced per
   band.

Every step produces exactly the same pixels, stars and masks as before; tests
compare the chunked results with whole-image references.

## Results

Same 48 MP photos, one process doing open (read and sky mask), detection with
photometry and a Result export with all effects; maximum resident memory.

| File | Before | After |
| --- | --- | --- |
| Uncompressed | 4.42 GB | 1.81-1.92 GB |
| Compressed (zlib) | 3.62-4.35 GB | 1.81-2.24 GB |

That is about 40-45 bytes per pixel instead of 75-90, of which 12 bytes are the
working image. Per stage, measured in separate processes:

| Stage | Before | After |
| --- | --- | --- |
| Thumbnail (compressed) | +822 MB | +91 MB |
| Read working image (compressed) | +1350 MB | +618 MB |
| Sky mask (compressed) | ~+700 MB | ~+370 MB |
| Detection + photometry, peak | ~+970 MB | ~+470 MB |
| Memory left after detection | ~+670 MB | ~+60 MB |
| Export | ~+1900 MB | ~+290 MB |

The exported file is byte-for-byte the same (identical pixel hash for both
files, old and new code). Times are unchanged within run-to-run noise.

## Not done yet

- The working image itself (12 bytes per pixel) still lives in RAM; keeping it
  in a smaller or tiled form is the next step for panoramas that do not fit.
- Detection's peak scales with the number of CPU cores (one ~40 MB tile per
  worker); fewer tiles in flight would save memory at a noticeable speed cost.
- Speed: a Result export with every effect on 48 MP still takes about 2.5
  minutes, and the multiscale detector about 30 seconds.
