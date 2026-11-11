# Image pipeline

## Input and color

Input validation rejects unsupported page count, channel layout, sample format, orientation and color profiles. Accepted image samples become immutable float32 linear-sRGB working pixels. RGB images with a matrix/TRC profile other than sRGB (for example Adobe RGB, Display P3 or ProPhoto) are converted on open: each channel's tone curve is decoded, a 3x3 matrix maps the profile's colorants to sRGB primaries (relative colorimetric, no black point compensation, as LittleCMS does for matrix/shaper profiles), and results outside sRGB are clipped to [0, 1]. Profiles with lookup tables or unknown tags are rejected. The sky mask and thumbnail see the same converted colors. The original file and its numerical samples remain the source of truth.

The detector uses quantized analysis proxies. Global normalization prevents each tile from independently ranking or rescaling its sky region. Approximate seed selection accelerates candidate qualification; it is not an exhaustive evaluation of every source configuration. The optional multiscale detector estimates local background and tests compact structure at several scales.

## Mask and measurement

The sky model reads encoded source samples. Tiled inference and overlap combination produce sky, foreground and uncertain labels. Foreground is excluded before source normalization and detection and masked again during rendering.

Photometry estimates local background planes and signed aperture flux on original pixels. Nearest-source ownership partitions crowded apertures. Measured FWHM and flux define a fixed preference weight for Soft filter. The additional source-peak metadata combines measured peak luminance with integrated source RGB. It is independent of the effect-color normalization.

## Render order

1. Derive bounded reduction of small measured source wings and expansion of larger source profiles when Soft filter is enabled.
2. Form the working base as `original - reduction + enlargement`.
3. Combine enabled Primary, Secondary, Soft Flare and Ring Flare layers with Screen composition. Each source's light is first limited to the display range along a path to white (hue and luminance kept, saturation reduced), on both CPU and GPU.
4. Compose those layers over the corrected base and convert to display or export encoding.

The CPU and GPU may calculate independent layer buffers in a different scheduling order; the final composition always follows the order above. Halo controls do not select or scale the source prefilter. Soft filter also biases the Soft Flare emission layer using the same fixed source-size preference: it suppresses small-source halos and strengthens large-source halos. Other emission groups remain unchanged. The catalogue is not detected again when a slider moves. Missing or uncertain measurements receive no source-profile correction.

Source corrections preserve the exact central sample of a centered source model and taper at their support boundary. Overlapping corrections of the same kind use a per-channel maximum, avoiding stacked subtraction. This is a digital approximation of source-wing editing, not a complete optical or deblending model.

## Regions and cancellation

Preview coordinates map to original pixels through an explicit origin and integer sampling step. The catalogue remains global when rendering a crop. Original view bypasses source corrections and effects.

Workers check cancellation between bounded batches or image strips. They cannot interrupt an individual native-library call. Export renders, composites, corrects, encodes and writes one 256-row band at a time, so only the working image stays full-size in memory; see [Memory and chunked processing](memory.md).
