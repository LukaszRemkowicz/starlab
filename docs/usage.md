# Usage

## Open and analyze

Use the folder button or pass a TIFF filename to `uv run star-effects-lab`. The file dialog remembers its last directory through Qt settings. A filename on the command line intentionally starts import automatically.

Import validates layout and color encoding before creating an immutable linear-RGB working image. A packaged ONNX model separates sky from foreground. Source detection then examines the full image using a global catalogue; selecting a preview region does not restrict detection to that region.

The mask button displays sky, foreground and uncertain regions. Mask visibility is independent of detection and rendering protection. Automatic detection can be adjusted in General. Reanalysis replaces manual catalogue edits and requires acknowledgement when edits would be lost.

## Effects

Primary and Secondary have separate controls under Spikes. Primary exposes the point count in addition to quantity, intensity, length, angle, sharpness, spread, saturation, hue and rainbow controls. Secondary uses a fixed angular pattern with its own quantity, intensity, geometry and color controls. Soft Flare and Ring Flare have independent radial-size, brightness, color and edge controls.

Quantity controls an effect's source-brightness cutoff; it is not a detection-count limit. Increasing it can admit fainter measured sources. Intensity changes emitted effect light. Saturation0 leaves the measured source color unchanged before compositing; positive values increase effect saturation. Very strong light can still clip in the displayed or exported result.

Soft filter is separate from halo selection. It reduces small source wings and expands larger source profiles on the original image before any effect layers are composited. Halo Quantity, Intensity, Size and color controls do not change this prefilter. It remains active at `Quantity = 0` / `Intensity = 0` while Soft Flare is enabled. The counter beside Enable soft flare reports halo selection, not the number of source corrections.

## Preview and manual edits

Use Pan, Edit, Add star and Hide/show to inspect or adjust the source catalogue. Undo/Redo tracks catalogue and parameter changes. Moving a measured source discards measurements from its previous position.

Original shows source pixels, Result includes source corrections and effect layers, Effects only shows added layers, and Compare splits the image. Hold Before temporarily restores the original view. 100% reads actual pixels of the visible source region. Fit trades detailed sampling for a bounded preview; it is unsuitable for judging individual-pixel sharpness.

## Save

Presets are JSON settings files. Export writes a new 16-bit TIFF, either the complete Result or added Effects only on black. Source corrections cannot be represented by the added-light layer alone. The input image is never overwritten. The export color encoding is sRGB for RGB images; arbitrary original metadata is not retained.
