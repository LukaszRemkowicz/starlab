# StarLab

StarLab is a standalone desktop editor for star diffraction and glow effects on TIFF photographs. It uses Python, Qt, Numba and OpenGL. The application currently opens as **Star Effects Lab** and runs through the `star-effects-lab` command.

## Run

Install [uv](https://docs.astral.sh/uv/), then run from the checkout:

```powershell
uv sync
uv run star-effects-lab
uv run star-effects-lab 'D:\photos\night-sky.tif'
```

An optional image argument opens that file at startup. Without it, use the folder button. The file dialog remembers its last directory. Python 3.12–3.14 is supported by the project metadata; uv selects an appropriate interpreter.

The current desktop preview requires OpenGL 3.3. This checkout has been exercised on Windows. macOS packaging and runtime verification are still pending.

## Editing

1. Open a supported TIFF. StarLab reads the original pixels, builds a sky mask and detects sources across the image.
2. Use **General** to adjust detection. The foreground is always excluded. The mask icon changes mask visibility; it does not disable protection.
3. Configure **Primary**, **Secondary**, **Soft Flare** and **Ring Flare**. Each group has independent controls. Advanced mode exposes color, rainbow and edge parameters; switching modes preserves values.
4. Use **Original**, **Result**, **Effects only** or **Compare**. **100%** displays original-resolution pixels for the visible region. Hold **Before** to inspect the source temporarily.
5. Export a new 16-bit TIFF. The application protects the input file from overwrite. Presets store effect settings; they do not store the image or manually edited source catalogue.

### Soft filter

The **Soft filter** control belongs to Soft Flare. It modifies measured source profiles before effect layers are composited: smaller stars shrink gently and larger stars expand. It works with halo **Quantity=0** and **Intensity=0**. These two controls govern the additional halo, not the source-profile correction.

```text
original pixels → source-profile correction → effect layers → result
```

`Soft filter = 0` preserves the source pixels. Disabling the whole Soft Flare group disables its source correction as well. Original always displays the unmodified source; Effects only displays added effect layers, excluding source-profile corrections. Source positions and classification stay fixed during slider movement.

## Supported images and limits

- Single-page, RGB or grayscale TIFF, 8 or 16 bits per sample; no alpha.
- RGB without an ICC profile is interpreted as sRGB. Untagged grayscale is interpreted as linear. Supported embedded profiles are checked numerically; an unsupported profile must be converted to sRGB in an image editor before import. Removing a profile is not a color conversion.
- The working image is immutable float32 linear RGB. Large panoramas need substantial RAM. TIFF orientation must already be normalized to 1.
- The sky model and detector can miss or misclassify sources. Crowded, saturated and irregular stars have less reliable photometry. Soft filter uses approximate source profiles, not full PSF fitting.
- Fit view samples a reduced grid; use 100% for sharpness and pixel alignment. Preview does not implement complete monitor-profile color management.
- No FITS, HDR workflow, batch export, persistent editing sessions or macOS installer. Interactive latency and startup depend on image content, hardware and first-use Numba compilation; there is no universal 10-second guarantee.

## Development

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

The full tests use checked-in fixtures and do not depend on an external research folder. `tools/`, local environments, IDE settings and generated artifacts are excluded from publication.

[Documentation](docs/README.md) · [Architecture](docs/architecture.md) · [Image pipeline](docs/image-pipeline.md) · [Assets](docs/assets.md) · [Development and verification](docs/development.md)
