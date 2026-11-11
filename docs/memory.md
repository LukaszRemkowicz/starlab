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
scope for now. Everything else is processed in row bands:

1. **Streaming decode.** Encoded samples are read band by band: memory-mapped
   when the file allows it, otherwise decoded segment by segment
   (`TiffPage.segments`). Reading the working image, the thumbnail and the sky
   mask never hold a whole decoded file. Nothing is written to disk.
2. **Streaming export.** Effects are rendered for one full-resolution row band
   at a time, composited with that band of the working image, corrected by the
   Soft filter, encoded to 16-bit and handed to the TIFF writer. No full-size
   effect layer, float result or 16-bit result exists. Output is bit-identical to
   rendering the whole image at once.
3. **Bounded detection.** Detection keeps a bounded number of tiles in flight and
   releases its temporary buffers when it finishes.

Not in scope yet: keeping the working image itself in a smaller or tiled form
(the remaining 12 B/px), and the speed of exports with many Secondary rays.

## Targets

- Peak for a 48 MP session (open, detect, export) about 1.2 GB for both
  uncompressed and compressed files, down from 2.5 and 4.2 GB.
- Exported pixels identical to the previous export.
- No step slower by more than 10%.
