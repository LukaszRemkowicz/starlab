# Architecture

```text
src/starlab/
  domain/          images, source catalogue, detection, photometry and effects
  application/     sessions, history, use cases and I/O interfaces
  infrastructure/  TIFF/JSON adapters, color tables, sky model and GPU renderer
  presentation/qt/ desktop controls, preview interaction and worker coordination
  bootstrap.py     dependency construction
```

## Boundaries

The domain implements image and source operations without importing Qt. Detection builds a global source catalogue; photometry measures original pixels rather than preview bitmaps. Effect profiles and source corrections have CPU implementations used by export and numerical verification.

The application layer owns use cases and immutable worker snapshots. Session is owned by the GUI thread and records settings/catalogue history without copying the full image. ImageServices coordinates image reading, mask construction, detection, rendering and export through injected adapters.

Infrastructure handles external formats, packaged numerical data, the ONNX sky model and OpenGL. TIFF readers map suitable files and convert blocks to the working representation. The GPU renderer evaluates preview profiles at a bounded output size and reports independent group and source-correction buffers.

The Qt layer connects user input to application operations. Long work runs outside the GUI thread. New preview requests supersede obsolete ones; revision checks reject stale completions. Catalogue changes and disabled groups clear obsolete image corrections before later zoom updates can display them.

## CPU and GPU

The interactive preview uses OpenGL 3.3. CPU export remains the numerical reference. Both paths apply source reduction/enlargement before Screen composition of effect layers. At reduced sampling, GPU radial glows integrate a pixel footprint whereas the CPU reference uses point samples; these reduced representations are not bit-identical.

Numba compiles hot CPU loops, including source qualification, catalogue joining, sample conversion and source-profile correction. The first use of a new specialization can incur compilation latency. The normal application does not persist calculated photo catalogues or masks across process launches. In-process preview buffers reuse unchanged layers during interaction.

## Current maintenance boundary

Complexity, annotation and public-documentation rules plus strict type checking now run in CI. Classes represent state or resource ownership; numerical kernels remain functions. Backend agreement is verified for recorded fixtures and one Windows GPU, not every possible image or driver.

## State ownership and numerical functions

Classes own meaningful state: `Session` owns editing history, `ImageFrame` owns
immutable working pixels, `SkyMaskService` owns a segmentation network, the Qt
worker owns job lifecycle and the GPU renderer owns its context/resources.
`ColorTransform` now owns the lazy universal color cube together with its
initialization lock. It publishes one validated, immutable table; failed loads
leave no partial state. This replaces independent mutable module globals, not
an image-result cache.

Typed immutable settings, source records and view snapshots describe values.
Pure numerical sampling, fitting, ray generation and JIT kernels remain functions;
putting them into stateless classes would add indirection without clarifying
ownership. Independent Python/Numba and CPU/GLSL implementations remain useful
numerical verification boundaries.
