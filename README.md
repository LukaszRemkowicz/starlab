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
- [Update or remove the app](#update-or-remove-the-app)
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
- **Soft filter.** Small stars get smaller and big stars larger, clearly
  visible at high values, much like a diffusion filter on the lens.
- **Color.** Keep each star's natural color, change the hue and saturation, or
  add a rainbow tint to the spikes.
- **Live preview.** See every change straight away, compare it with the
  original, and zoom in to check single pixels.
- **Fix the detection by hand.** Move, add, hide or recolor individual stars.

## Before you start

You need:

- **A computer with Windows or macOS.** Both have been tested. Linux may work
  but has not been checked.
- **A graphics card that supports OpenGL 3.3.** Almost any graphics card from
  the last ten years does.
- **Enough memory for your photo.** The app keeps one working copy of the photo
  in memory and processes the rest in pieces. Measured on a 48-megapixel photo,
  it needs about 2 GB at its peak, compressed or not. That is roughly 40-45
  bytes per pixel, so a 200-megapixel panorama needs about 8-9 GB.
- **An internet connection** for the first install.

## Install

Star Effects Lab does not have an installer yet. You install it with
[uv](https://docs.astral.sh/uv/), a small tool that downloads the right Python
version and everything else the app needs. You do not have to install Python
yourself.

1. **Install uv.** On Windows, open PowerShell and run:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   On macOS, open Terminal and run:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Close the window and open a new one so it finds the `uv` command. The commands
   below are the same on Windows and macOS. For Linux, see the
   [uv install guide](https://docs.astral.sh/uv/getting-started/installation/).

2. **Download Star Effects Lab.** If you have Git:

   ```powershell
   git clone --branch dev https://github.com/LukaszRemkowicz/starlab.git
   cd starlab
   ```

   Without Git, switch the branch on the GitHub page to **dev**, click **Code →
   Download ZIP**, unpack the archive, and open PowerShell or Terminal in the
   unpacked folder.

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

On macOS, a path looks like `~/Pictures/night-sky.tif`.

The first start after installing or updating prepares the app's image code in
the background, which takes up to about a minute once. The window is usable
straight away; a photo you open during that time simply waits for it. After that,
finding stars usually takes about a second.

## Update or remove the app

To update, open PowerShell or Terminal in the `starlab` folder and run:

```powershell
git pull
uv sync
```

Without Git, download the ZIP again and replace the folder. To remove the app,
delete the `starlab` folder and the app's data folder (the folder that holds
`logs`, listed under [Troubleshooting](#troubleshooting)).

## Your first edit in five steps

1. **Open a photo.** Click **Load Image** or drag a TIFF into the window. Later
   you can open another photo with the folder icon (Ctrl+O, or Cmd+O on a Mac). The
   app builds the sky mask and finds the stars. The counter in **General**
   shows how many stars it found.
2. **Pick a starting look.** At the bottom, choose a preset: **Default
   Spikes**, **Subtle Glow** or **Dramatic Diffraction**.
3. **Adjust the effects.** Open the **Spikes**, **Soft Flare** and **Ring Flare**
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
| **Minimum brightness** / **Detection threshold** | How bright a star must be to be found. Lower values find more, fainter stars. The original-style detector uses a brightness from 0.04 to 0.98; the multiscale detector uses a threshold in σ (1-30) above the local noise. |
| **Detect stars** | Runs detection again with the current settings. If you have edited stars by hand, the app asks first, because detection replaces those edits. |
| **Show star centers** | Marks each detected star in the preview. |

The app remembers the stars it found, and your edits to them, only while the
photo is open. Moving sliders never runs detection again. When you close the app
or open another photo, the stars are found afresh; nothing is saved next to your
files.

### Effect tabs

Each effect tab (**Primary** and **Secondary** under **Spikes**, plus **Soft
Flare** and **Ring Flare**) has its own **Enable** checkbox, its own sliders,
and a **Reset this group** button.

The panel opens in **Advanced** mode, which shows every slider. Choose **Basic**
to see the main sliders only. Switching between the two keeps your values.

| Slider | What it does |
| --- | --- |
| **Quantity** | Which stars get the effect. Higher values add it to fainter stars too; lower values keep it to the brightest stars. |
| **Intensity** | How bright the effect is. |
| **Length** | How long the spikes are. |
| **Angle** | Rotates the spikes. |
| **Number of points** | How many spikes each star gets (Primary only). |
| **Spike sharpness**, **Spike spread** | How thin and how fanned out the spikes are. |
| **Size** | How large the glow or ring is. |
| **Soft filter** | Shrinks small stars and enlarges big ones, noticeably at high values (Soft Flare tab). At 0 the stars stay as they are. |
| **Edge falloff**, **Inner/Outer edge width and falloff** | How softly the glow or ring fades at its edges. |
| **Color saturation**, **Color hue** | Change the effect's color. At saturation 0 each effect keeps its star's natural color. |
| **Rainbow intensity**, **frequency**, **phase** | Add a rainbow tint along the spikes and control how it repeats. |

**Soft filter** follows the other **Soft Flare** controls: **Quantity** picks
which stars it changes, **Size** and **Edge falloff** set how wide and how soft
the enlarged halo is, and **Saturation** and **Hue** tint it. **Intensity** only
changes the glow. So you can use Soft filter on its own: set **Intensity** to 0
in **Soft Flare** and keep **Soft filter** above 0. With **Quantity** at 0 it
changes no stars. Turning off the whole **Soft Flare** tab also
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

The mask icon in the toolbar switches the sky-mask colors on and off over the
current view. The foreground is always protected, even when the mask is hidden.

## Fixing individual stars

Pick a tool in the toolbar:

- **Pan image** moves around the photo. You can also hold **Space** with any
  tool to pan.
- **Edit star** opens a star's position, brightness and color when you click
  it.
- **Add star** places a new star where you click.
- **Hide / show star** removes a star from the effects without deleting it.

**Undo** (Ctrl+Z, or Cmd+Z on a Mac) and **Redo** (Ctrl+Y, or Cmd+Shift+Z on a
Mac) work for star edits and for slider changes. One slider drag counts as one
step.

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

The export is always a new 16-bit TIFF. Color photos are saved in sRGB with an
embedded sRGB profile; grayscale photos are saved with the same tone curve and
no profile. The app will not let you overwrite the photo you opened. Camera data
such as EXIF is not copied into the export.

## Which images work

| Works | Does not work yet |
| --- | --- |
| TIFF (`.tif`, `.tiff`), 8 or 16 bits per channel | JPEG, PNG, RAW, FITS; 12-bit or 32-bit TIFF |
| Uncompressed or compressed (LZW, ZIP) | CMYK, palette images |
| RGB color with an sRGB profile, no profile, or a matrix/curve profile such as Adobe RGB, Display P3 or ProPhoto (converted to sRGB on open) | RGB profiles built on lookup tables |
| RGB or grayscale with an extra alpha channel (ignored; premultiplied colors are restored, fully transparent areas become black) | Premultiplied alpha in planar TIFFs |
| Grayscale with no profile | Grayscale with an embedded profile (such as Gray Gamma 2.2) |
| Single-page TIFF | Multi-page TIFF, HDR images |

Files without a profile, color or grayscale, are read with the sRGB tone curve,
the way most editors save them.

If the app rejects your file, its message tells you why. The usual fixes, all
done in your photo editor:

- **Unsupported color profile:** convert a color image to sRGB. Removing the
  profile is not the same as converting it. Common wide-gamut profiles are
  converted for you on open; colors outside sRGB are clipped. For grayscale, convert to sRGB grayscale or
  save without a profile.
- **Rotated image:** save the image with its rotation applied.

## Tips

- **Start from a finished photo.** The app expects an image that is already
  stretched and processed for viewing, not a raw linear stack.
- **Too many spikes?** Lower **Quantity** so only the brightest stars get them.
- **Effects on the landscape?** Look at the **Sky mask** view. Areas marked red
  are protected; if the mask is wrong in a spot, hide the affected stars by hand.
- **Judge sharpness at 100%,** not in Fit view.
- **Very bright spikes fade to white.** Where an effect is brighter than the
  image can show, its color thins out toward white, like an overexposed star,
  while the edges keep their color. Lower **Intensity** if you want more color
  in the brightest parts.

## Troubleshooting

**The window is black or the app closes right after starting.**
Update your graphics card driver. The preview needs OpenGL 3.3.

**Finding stars takes up to a minute the first time.**
This happens once after installing or updating, while the app prepares its image
code in the background. Later, detection usually takes about a second. Large
panoramas need more time and memory, so close other memory-hungry programs.

**The app says it cannot start because of log files.**
The app always writes a log and refuses to start if it cannot. Check that the
folder named in the message exists and that you can write to it; the usual
locations are listed below.

**A star was missed, or something that is not a star got an effect.**
Use **Add star** or **Hide / show star**, or change the detection threshold in
**General**.

**The app crashed.**
The app writes logs you can attach to a bug report. They are usually here:

| System | Log file |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Star Effects Lab\Star Effects Lab\logs\startup.log` |
| macOS | `~/Library/Application Support/Star Effects Lab/Star Effects Lab/logs/startup.log` |

In the same folder, `native-crash.log` records low-level crashes and
`preparation.log` records problems while the app prepares its image code. The
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
