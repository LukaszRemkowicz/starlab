# Star Effects Lab

Star Effects Lab adds diffraction spikes, soft glow and halo rings to the stars
in your night-sky photos. Open a TIFF and the app finds the stars for you. You
then shape the effects with sliders and export a new 16-bit TIFF. Your original
file is never changed.

The app is free for everyone under the [MIT License](LICENSE).

- [What it does](#what-it-does)
- [Before you start](#before-you-start)
- [Install](#install)
- [Start the app](#start-the-app)
- [Your first edit in five steps](#your-first-edit-in-five-steps)
- [The controls](#the-controls)
- [Viewing your image](#viewing-your-image)
- [Fixing individual stars](#fixing-individual-stars)
- [Presets](#presets)
- [Exporting](#exporting)
- [Which images work](#which-images-work)
- [Tips](#tips)
- [Troubleshooting](#troubleshooting)
- [For developers](#for-developers)

## What it does

- **Finds stars automatically.** A built-in neural network separates the sky
  from trees, mountains and buildings. Effects are added only to stars in the
  sky, never to the landscape.
- **Diffraction spikes.** Choose the number of spikes, their length and angle,
  and add a second, finer set of spikes if you want one.
- **Soft glow and halo rings.** Add the look of a softening filter or a lens
  ring around bright stars.
- **Soft filter.** Small stars get slightly smaller and big stars slightly
  larger, much like a diffusion filter on the lens.
- **Color.** Keep each star's natural color, change the hue and saturation, or
  add a rainbow tint to the spikes.
- **Live preview.** See every change straight away, compare it with the
  original, and zoom in to check single pixels.
- **Fix the detection by hand.** Move, add, hide or recolor individual stars.

## Before you start

You need:

- **A computer with Windows.** That is where the app has been tested. It may
  run on macOS or Linux, but nobody has checked that yet.
- **A graphics card that supports OpenGL 3.3.** Almost any graphics card from
  the last ten years does.
- **Enough memory for your photo.** As a rough guide, plan on at least 12 bytes
  of RAM per pixel. A 24-megapixel photo needs at least 300 MB; a large panorama
  can need several gigabytes.
- **An internet connection** for the first install.

## Install

Star Effects Lab does not have an installer yet. You install it with
[uv](https://docs.astral.sh/uv/), a small tool that downloads the right Python
version and everything else the app needs. You do not have to install Python
yourself.

1. **Install uv.** Open PowerShell and run:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Close PowerShell and open it again so it finds the `uv` command. On macOS or
   Linux, follow the [uv install guide](https://docs.astral.sh/uv/getting-started/installation/)
   instead.

2. **Download Star Effects Lab.** If you have Git:

   ```powershell
   git clone --branch dev https://github.com/LukaszRemkowicz/starlab.git
   cd starlab
   ```

   Without Git, switch the branch on the GitHub page to **dev**, click **Code →
   Download ZIP**, unpack the archive, and open PowerShell in the unpacked
   folder.

3. **Install what the app needs.** This takes a few minutes the first time:

   ```powershell
   uv sync
   ```

## Start the app

From the `starlab` folder, run:

```powershell
uv run star-effects-lab
```

To open a photo straight away, add its path:

```powershell
uv run star-effects-lab 'D:\photos\night-sky.tif'
```

The first start, and the first time you analyze an image, can be slow while the
app prepares its fast image code. Later starts are quicker.

## Your first edit in five steps

1. **Open a photo.** Click **Load Image** or drag a TIFF into the window. The
   app builds the sky mask and finds the stars. The counter in **General**
   shows how many stars it found.
2. **Pick a starting look.** At the bottom, choose a preset: **Default
   Spikes**, **Subtle Glow** or **Dramatic Diffraction**.
3. **Adjust the effects.** Open the **Spikes**, **Soft flare** and **Ring flare**
   tabs and move the sliders. The preview updates as you drag.
4. **Check the result.** Hold **Hold for Before** to see the original for a
   moment, or switch the view to **Compare**. Click **100%** to inspect fine
   detail.
5. **Export.** Click **Export**, choose **Result**, and pick a file name.

## The controls

The panel on the right has one tab for star detection and one tab for each
effect.

### General: finding stars

| Control | What it does |
| --- | --- |
| **Automatic Star Detection** | Finds stars as soon as you open an image. Turn it off if you want to start detection yourself. |
| **Star detector** | **Original-style detector** suits most photos. **Multiscale · larger stars** also picks up big, soft stars. |
| **Minimum brightness** / **Detection threshold** | How bright a star must be to be found. Lower values find more, fainter stars. |
| **Detect stars** | Runs detection again with the current settings. If you have edited stars by hand, the app asks first, because detection replaces those edits. |
| **Show star centers** | Marks each detected star in the preview. |

### Effect tabs

Each effect tab (**Primary** and **Secondary** under **Spikes**, plus **Soft
flare** and **Ring flare**) has its own **Enable** checkbox, its own sliders,
and a **Reset this group** button.

Choose **Basic** to see the main sliders only, or **Advanced** to see all of
them. Switching between the two keeps your values.

| Slider | What it does |
| --- | --- |
| **Quantity** | Which stars get the effect. Higher values add it to fainter stars too; lower values keep it to the brightest stars. |
| **Intensity** | How bright the effect is. |
| **Length** | How long the spikes are. |
| **Angle** | Rotates the spikes. |
| **Number of points** | How many spikes each star gets (Primary only). |
| **Spike sharpness**, **Spike spread** | How thin and how fanned out the spikes are. |
| **Size** | How large the glow or ring is. |
| **Soft filter** | Shrinks small stars and enlarges big ones (Soft flare tab). At 0 the stars stay as they are. |
| **Edge falloff**, **Inner/Outer edge width and falloff** | How softly the glow or ring fades at its edges. |
| **Color saturation**, **Color hue** | Change the effect's color. At saturation 0 each effect keeps its star's natural color. |
| **Rainbow intensity**, **frequency**, **phase** | Add a rainbow tint along the spikes and control how it repeats. |

**Soft filter** works even when the glow itself is set to zero, so you can use
it on its own: set **Quantity** or **Intensity** to 0 in **Soft flare** and
keep **Soft filter** above 0. Turning off the whole **Soft flare** tab also
turns Soft filter off.

## Viewing your image

Use the menu in the toolbar to switch between views:

| View | What you see |
| --- | --- |
| **Result** | Your photo with all effects. |
| **Original** | Your photo exactly as it was opened. |
| **Effects only** | Just the added spikes and glow, on black. |
| **Compare** | A split view: original on one side, result on the other. |
| **Sky mask** | Blue for sky, red for foreground, yellow where the app is unsure. |

- **Zoom** with the mouse wheel or the **−** and **+** buttons.
- **Fit** shows the whole image; it is a quick, reduced preview.
- **100%** shows real pixels. Use it to judge sharpness.
- **Hold for Before** shows the original while you hold it.
- **Apply** redraws the preview with the current settings.
- **Cancel** stops a long task, such as detection.

The foreground is always protected, even when the mask is hidden.

## Fixing individual stars

Pick a tool in the toolbar:

- **Pan image** moves around the photo. You can also hold **Space** with any
  tool to pan.
- **Edit star** opens a star's position, brightness and color when you click
  it.
- **Add star** places a new star where you click.
- **Hide / show star** removes a star from the effects without deleting it.

**Undo** (Ctrl+Z) and **Redo** work for star edits and for slider changes. One
slider drag counts as one step.

## Presets

- **Default Spikes**, **Subtle Glow** and **Dramatic Diffraction** are built in.
  **Reset** goes back to Default Spikes.
- Use the folder and disk icons next to the preset menu to load and save your
  own presets as `.json` files.
- A preset stores only the effect settings. It does not store the photo or any
  stars you edited by hand.

## Exporting

Click **Export** and choose what to save:

- **Result:** your photo with all effects.
- **Effects only (added glow on black):** just the effects, ready to blend in
  another editor, for example with the Screen blend mode. The Soft filter
  changes to the stars themselves are not part of this layer.

The export is always a new 16-bit TIFF in sRGB color. The app will not let you
overwrite the photo you opened. Camera data such as EXIF is not copied into the
export.

## Which images work

| Works | Does not work yet |
| --- | --- |
| TIFF, 8 or 16 bits per channel | JPEG, PNG, RAW, FITS |
| RGB color or grayscale | Images with transparency (alpha), CMYK, palette images |
| sRGB color profile, or no profile | Other profiles, such as Adobe RGB or ProPhoto |
| Single-page TIFF | Multi-page TIFF, HDR or 32-bit images |

If the app rejects your file, its message tells you why. The usual fixes, all
done in your photo editor:

- **Wrong color profile:** convert the image to sRGB. Removing the profile is
  not the same as converting it.
- **Rotated image:** save the image with its rotation applied.
- **Transparency:** flatten the image before saving.

## Tips

- **Start from a finished photo.** The app expects an image that is already
  stretched and processed for viewing, not a raw linear stack. Files without a
  color profile are read as sRGB.
- **Too many spikes?** Lower **Quantity** so only the brightest stars get them.
- **Effects on the landscape?** Look at the **Sky mask** view. Areas marked red
  are protected; if the mask is wrong in a spot, hide the affected stars by hand.
- **Judge sharpness at 100%,** not in Fit view.
- **Very bright stars can clip** to pure white in the result. Lower
  **Intensity** if highlights look burnt out.

## Troubleshooting

**The window is black or the app closes right after starting.**
Update your graphics card driver. The preview needs OpenGL 3.3.

**Detection or export is very slow.**
The first analysis after starting the app is slower than later ones. Large
panoramas need time and memory; close other memory-hungry programs.

**A star was missed, or something that is not a star got an effect.**
Use **Add star** or **Hide / show star**, or change the detection threshold in
**General**.

**The app crashed.**
The app writes a log you can attach to a bug report. On Windows it is usually
here:

```text
%LOCALAPPDATA%\Star Effects Lab\Star Effects Lab\logs\startup.log
```

A separate `native-crash.log` in the same folder records low-level crashes. The
logs contain no image data and are never sent anywhere.

## For developers

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pylint src/starlab
```

More detail is in [docs](docs/README.md): [architecture](docs/architecture.md),
[image pipeline](docs/image-pipeline.md), [assets](docs/assets.md) and
[development](docs/development.md).

## License

[MIT](LICENSE). You may use, change and share Star Effects Lab freely, including
for commercial work.
