# Packaged assets

Runtime data is checked in beside the code that consumes it. These files are not saved analysis results for a user's photograph.

| Asset | Consumer | Purpose |
|---|---|---|
| `src/starlab/domain/detection/analysis_luts.npz` | `domain/detection/proxy.py` | Numeric read/write lookup arrays used by analysis-proxy sample conversion. |
| `src/starlab/domain/detection/full_lcms_lut.npz` | `domain/detection/proxy.py` | Compressed, lossless lookup data for the RGB8 color transform. |
| `src/starlab/domain/detection/full_lcms_lut.json` | `domain/detection/proxy.py` | Transform description, dimensions, encoding, library versions and integrity hashes for that archive. |
| `src/starlab/infrastructure/models/sky-mask.onnx` | `infrastructure/sky_mask.py` | Trained sky/foreground segmentation network. |
| `src/starlab/infrastructure/models/sky-mask.json` | `infrastructure/sky_mask.py` | Network identity, preprocessing, tiling and recorded validation metadata. |
| `src/starlab/infrastructure/shaders/preview.vert` | `infrastructure/gpu_preview.py` | Positions per-source geometry for OpenGL preview drawing. |
| `src/starlab/infrastructure/shaders/preview.frag` | `infrastructure/gpu_preview.py` | Evaluates per-pixel rays, glows and source-profile corrections. |

NPZ is a NumPy archive of named arrays. The JSON file describes its corresponding data and does not replace it. ONNX stores the neural-network graph and weights. GLSL shader sources run on the GPU; they are executable parts of the preview, not documentation or Python modules.

`analysis_luts.npz` is 5,624 bytes: `read` is 256 float32 values and `write` is 4,096 uint8 values (5,120 array bytes in total). The loader verifies the archive SHA-256 before exposing read-only arrays. READ approximates gamma 2.4 but differs from a freshly calculated power by up to about 1.13e-6; retaining stored values preserves threshold decisions. WRITE quantizes normalized analysis levels back to byte codes. Neither array is a source-detection result or a cached photograph.

The RGB8 transform archive covers 256×256×256 input colors and reconstructs a uint8 three-channel table from compressed residuals. The loader verifies the compressed archive hash, metadata and array dimensions/dtypes, then the reconstructed-table hash. The archive is236,561 bytes, whereas the resulting table is50,331,648 bytes (48MiB), plus three256-byte baseline arrays during reconstruction. JSON also records the historical generator/adapter identity and library versions; these provenance fields do not execute or replace the transform. See the JSON metadata for the exact encoding and parameters. The sky model likewise validates its ONNX SHA-256 against recorded metadata before use.

## Test fixtures

`tests/fixtures/detection/` contains six small detector input images and three numerical reference catalogues. `tests/fixtures/rendering/` contains 42 expected effect images plus `cases.json`, which records the catalogue and settings for each image.

Published TIFF fixtures retain their original pixel arrays and embedded color profile where present. Incidental TIFF metadata is omitted. Rendering tests consume plain JSON inputs directly; no descriptor parser or offline investigation scripts are required. Historical reports, screenshots, binary descriptor dumps and large generated catalogues are excluded from publication.
