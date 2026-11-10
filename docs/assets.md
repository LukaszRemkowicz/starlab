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

## Secondary appearance pattern

`SECONDARY_RAYS` is one ordered table of84 `(angle, weight)` pairs in
`domain/effects/profiles.py`. Angle units are0.1degree; weights are integer
percent before conversion to relative amplitudes. Rotation shifts angles and
preserves weights/order. CPU geometry and GPU instance counts derive from this
table; the shader's uniform arrays provide capacity for84 directions. This
pattern controls the look of Secondary rays. It is independent of the detector's
eight-direction sampling stencil, and is not a polymorphic abstraction.

## GPU shader stages

The vertex shader expands one measured source into ray/radial quads, rotates
ray directions and maps full-image pixel centers to viewport coordinates.
Uniform `kind` selects Primary0, Secondary1, Soft2, Ring3, source shrink4 or
source grow5. The84-direction array capacity is distinct from the active count.

The fragment shader evaluates transverse antialiasing and longitudinal ray
falloff, radial halo/ring profiles, optional rainbow modulation, or source-profile
correction magnitudes. Pixel-center and image-y/OpenGL-y conventions are explicit
in the source. Reduced radial previews integrate64 subpixel positions to avoid
missing narrow emitters; full-resolution evaluation uses the native point grid.

The Python renderer owns GL resources, uniforms, draw order and blend modes.
It combines source corrections before added effect layers. Shader output alone
is therefore not a finished photograph. CPU export uses independent NumPy/Numba
profiles; duplicating the mathematics across execution languages is intentional.
Numerical GPU tests must compile and execute both shader files on a real context;
CPU fixture tests alone cannot detect a broken uniform or shader change.
