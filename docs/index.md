# Open-AISP

Open-AISP is an open-source AI-ISP playground for building an end-to-end image signal processing pipeline. It starts from synthetic raw degradation and extends toward neural reconstruction, HDR synthesis, tone mapping, and diffusion-based image enhancement.

The project is organized into five modules:

| Module | Status | Purpose |
| --- | --- | --- |
| `raw-sim` | Active | Converts sRGB images into noisy packed RAW and clean linear RGB supervision. |
| `JDD` | Active | Restores clean linear RGB from noisy RAW bursts with joint denoising and demosaicing. |
| `HDR` | Placeholder | Future multi-frame high dynamic range synthesis from exposure brackets. |
| `AITM` | Placeholder | Future AI tone mapping from linear/HDR images to display-ready images. |
| `DiffIPE` | Placeholder | Future diffusion-based image post-enhancement. |

The two implemented modules are intentionally coupled through a single degradation interface: `JDD` calls `raw-sim` to produce RAW training samples. When the camera model, CFA layout, PSF blur, or noise model changes in `raw-sim`, the JDD data pipeline follows the same behavior.

## Pipeline Overview

The current training flow is:

```text
sRGB image
  -> raw-sim camera degradation
  -> noisy packed RAW burst
  -> JDD restoration network
  -> clean linear RGB
```

`raw-sim` is responsible for physically motivated degradation. `JDD` is responsible for burst restoration from noisy RAW to linear RGB.

## Languages

English is the default documentation language. A Chinese version is available from the language selector.
