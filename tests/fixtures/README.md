# Regression fixtures

The tests require only these checked-in files and runtime package assets.

- `detection/`: six input TIFFs and three numeric catalogues for source-location, color, tile-boundary and threshold regressions.
- `rendering/`: 42 expected TIFFs and a JSON manifest of source records and effect settings.

TIFF pixel arrays and embedded color profiles are preserved. Incidental descriptive metadata and external investigation-script dependencies are excluded. Do not regenerate expected pixels merely to make a changed algorithm pass: investigate and document intentional numerical changes first.

Intentional change: `color-quantity-1793869750289/primary-h50-s50.tif` and `primary-hm50-s50.tif` were regenerated when per-source light switched from per-channel clipping to a path to white (`render.fit_to_white`). Only the eight brightest spike-core pixels of each image changed; all other fixtures are unchanged.
