# raw-sim

`raw-sim` simulates the early camera pipeline in reverse: it starts from a clean sRGB image and produces noisy packed RAW plus a clean linear RGB target. This module is the degradation source used by downstream learning modules such as JDD.

## sRGB to Noisy RAW Pipeline

For each input sRGB image or crop, `raw-sim` applies:

```text
sRGB
  -> inverse gamma
  -> inverse CCM
  -> inverse AWB
  -> lens PSF / filter
  -> sensor level mapping
  -> CFA sampling and packing
  -> Poisson-Gaussian noise
  -> noisy packed RAW
```

The clean target is the linear RGB image after inverse gamma. Lens blur, CFA sampling, and sensor noise only affect the RAW branch.

## 1. Inverse Gamma

The input image is normalized to $[0, 1]$. The clean linear RGB target is computed with:

$$
I_{lin} = \operatorname{clip}(I_{srgb}, 0, 1)^\gamma
$$

The default examples use $\gamma = 2.2$.

## 2. CCM

The camera JSON stores a color correction matrix that maps camera RGB to linear RGB:

$$
I_{lin} = I_{cam} C^\top
$$

To unprocess sRGB into camera RGB, `raw-sim` applies the inverse transform:

$$
I_{cam}' = I_{lin}(C^{-1})^\top
$$

where $C$ is `ccm.matrix`.

## 3. AWB

White balance is inverted by dividing camera RGB by the configured per-channel gains:

$$
I_{cam,c} = \frac{I_{cam,c}'}{g_c}, \quad c \in \{R,G,B\}
$$

The configuration fields are `awb.red_gain`, `awb.green_gain`, and `awb.blue_gain`.

## 4. PSF Blur and Guided Filter

The `lens_psf` block is applied in camera RGB before sensor mapping and CFA sampling.

Gaussian blur uses a normalized Gaussian kernel:

$$
K(x,y) = \frac{1}{Z}\exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
$$

Example:

```json
{
  "lens_psf": {
    "enabled": true,
    "kernel": "gaussian",
    "kernel_size": 5,
    "sigma": 1.0
  }
}
```

Guided filtering is also supported:

```json
{
  "lens_psf": {
    "enabled": true,
    "kernel": "guided",
    "kernel_size": 5,
    "eps": 0.001
  }
}
```

For each local window $\omega_k$, guided filtering assumes:

$$
q_i = a_k G_i + b_k, \quad i \in \omega_k
$$

with:

$$
a_k = \frac{\operatorname{cov}_{\omega_k}(G, p)}
{\operatorname{var}_{\omega_k}(G) + \epsilon},
\qquad
b_k = \bar{p}_k - a_k \bar{G}_k
$$

`raw-sim` uses luma as the guide:

$$
G = 0.299R + 0.587G + 0.114B
$$

This provides edge-aware smoothing compared with pure Gaussian blur.

## 5. Sensor Level Mapping

Camera RGB is mapped into the sensor digital range:

$$
S = B + I_{cam}(W - B)
$$

where $B$ is black level and $W$ is white level after normalization by the maximum DN:

$$
DN_{max} = 2^{bitdepth} - 1
$$

If `black_level` is `"auto"`, the default 10-bit reference is 32 DN:

| Bit depth | Auto black level |
| --- | --- |
| 10 | 32 |
| 12 | 128 |
| 14 | 512 |

If `quantize` is enabled, values are rounded to the configured bit depth.

## 6. CFA Sampling

`raw-sim` supports Bayer/binning 2x2 packing and Quad Bayer 4x4 packing.

### Bayer and 2x2 Binning

For `RGGB`, the 2x2 tile is:

```text
R   Gr
Gb  B
```

Packed RAW has shape:

$$
H \times W \times 3 \rightarrow \frac{H}{2} \times \frac{W}{2} \times 4
$$

The packed channel order is:

```text
[R, Gr, Gb, B]
```

For `cfa.type = "binning"`, each packed channel can be locally filtered by a 2x2 window. `binning_operation: "average"` keeps the signal scale comparable to full readout, while `"sum"` accumulates the local signal.

### Quad Bayer 4x4

Quad Bayer repeats each Bayer color into a 2x2 same-color block. For `RGGB`, the 4x4 tile is:

```text
R   R   Gr  Gr
R   R   Gr  Gr
Gb  Gb  B   B
Gb  Gb  B   B
```

Packed RAW has shape:

$$
H \times W \times 3 \rightarrow \frac{H}{4} \times \frac{W}{4} \times 16
$$

The channel order is raster order over the 4x4 tile: `ch0` to `ch15`.

## 7. Poisson-Gaussian Noise

The active signal subtracts black level and normalizes by the active sensor range:

$$
x = \operatorname{clip}\left(\frac{S - B}{W - B}, 0, 1\right)
$$

The per-channel noise variance is:

$$
\sigma^2(x, c) = K_c x + \sigma_{read,c}^2
$$

The noisy RAW value is sampled with a Gaussian approximation to Poisson-Gaussian noise:

$$
\tilde{S}_c = S_c + \mathcal{N}\left(0, \sigma^2(x,c)\right)
$$

and then clipped to $[0,1]$.

### Analog Gain and ISO

The simulator defines:

$$
ISO = 100 \cdot AG
$$

where $AG$ is `camera.analog_gain`.

Noise calibration is stored at `base_analog_gain`. Runtime parameters are scaled by:

$$
r = \frac{AG}{AG_{base}}
$$

$$
K_c(AG) = K_{c,base} \cdot r^{\alpha_{shot}}
$$

$$
\sigma_{read,c}(AG) = \sigma_{read,c,base} \cdot r^{\alpha_{read}}
$$

The exponents are `shot_noise_gain_exponent` and `read_noise_gain_exponent`.

For binned readout, `raw-sim` also adjusts the effective shot and read noise according to the binning type and whether the binned value is averaged or summed.

## Batch Interface

The public batch interface is:

```python
from raw_sim.batch import srgb_to_raw_burst_batch

x, y = srgb_to_raw_burst_batch(
    rgb=batch_rgb,
    camera=camera,
    analog_gain=analog_gain,
    seed=seed,
    frames=3,
    noise_map_reduce="mean",
)
```

It returns:

- `x`: noisy RAW burst and noise map, shape `(B, frames * raw_channels + 1, H/scale, W/scale)`.
- `y`: clean linear RGB target, shape `(B, 3, H, W)`.

JDD uses this interface directly, so updates to `raw-sim` degradation are shared by training.
