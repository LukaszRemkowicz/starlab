# Regression fixtures

The tests require only these checked-in files and runtime package assets.

- `detection/`: six input TIFFs and three numeric catalogues for source-location, color, tile-boundary and threshold regressions.
- `rendering/`: 42 expected TIFFs and a JSON manifest of source records and effect settings.

TIFF pixel arrays and embedded color profiles are preserved. Incidental descriptive metadata and external investigation-script dependencies are excluded. Do not regenerate expected pixels merely to make a changed algorithm pass: investigate and document intentional numerical changes first.
