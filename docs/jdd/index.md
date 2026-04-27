# JDD

JDD is the joint denoise and demosaic module. It learns the mapping from noisy packed RAW bursts to clean linear RGB:

$$
f_\theta: \{R_t^{noisy}\}_{t=1}^{T}, M_\sigma \rightarrow I_{lin}
$$

where $T=3$ in the current training setup, $R_t^{noisy}$ are independently degraded RAW frames, and $M_\sigma$ is a noise standard-deviation map produced by `raw-sim`.

## Task Definition

The input is a packed RAW burst plus one noise-map channel:

$$
X = \operatorname{concat}(R_1, R_2, R_3, M_\sigma)
$$

For Bayer or 2x2 binning RAW, each frame has 4 packed channels, so:

$$
C_{in}=3 \times 4 + 1 = 13
$$

For Quad Bayer RAW, each frame has 16 packed channels:

$$
C_{in}=3 \times 16 + 1 = 49
$$

The target is clean linear RGB at the full spatial resolution:

$$
Y = I_{lin} \in [0,1]^{3 \times H \times W}
$$

## Random Degradation

JDD does not store a fixed noisy dataset by default. During training, it samples RGB patches and generates RAW degradation online through the public `raw-sim` batch interface.

For each sample:

1. Randomly choose an RGB source image.
2. Randomly crop a patch with size `data.patch_size`.
3. Sample analog gain from the camera module.
4. Convert sRGB to clean linear RGB target.
5. Run the `raw-sim` pipeline to produce three noisy RAW frames.
6. Build a noise map from the same calibrated noise model.

This makes every epoch see different crop positions, noise realizations, and analog gain values.

## Analog Gain Sampling

The camera JSON can define `camera.analog_gain` as a scalar, a two-element range, or an object with `min` and `max`. JDD uses `camera.analog_gain_sampling` to choose the sampling distribution.

Uniform sampling:

$$
AG = AG_{min} + u(AG_{max} - AG_{min}), \quad u \sim \mathcal{U}(0,1)
$$

Log-uniform sampling:

$$
AG = \exp\left(\log AG_{min} + u(\log AG_{max}-\log AG_{min})\right)
$$

The current ISO convention is inherited from `raw-sim`:

$$
ISO = 100 \cdot AG
$$

Because the Poisson-Gaussian noise model scales with analog gain, random gain sampling exposes the model to a broad noise range instead of a single fixed ISO.

## Burst Image Restoration

For each RGB patch, JDD synthesizes three noisy RAW frames from the same clean signal:

$$
R_t^{noisy} = \mathcal{D}(I_{srgb}; AG, \epsilon_t), \quad t \in \{1,2,3\}
$$

where $\mathcal{D}$ is the `raw-sim` degradation pipeline and $\epsilon_t$ is the frame-specific random noise seed.

The three frames share the same crop, camera parameters, analog gain, CFA layout, and lens filter, but use independent noise samples. This trains the network to use burst redundancy for denoising while also learning demosaicing and RAW-to-linear-RGB reconstruction.

## Training Data Path

When `data.simulate_on_device` is enabled, the DataLoader only returns:

- RGB patch
- random seed
- sampled analog gain
- metadata

The training loop then calls:

```python
from raw_sim.batch import simulate_burst_batch_on_device

x, y = simulate_burst_batch_on_device(
    rgb=batch["rgb"],
    analog_gain=batch["analog_gain"],
    seed=batch["seed"],
    camera=camera,
    frames=3,
    noise_map_reduce="mean",
)
```

This keeps the JDD degradation path coupled to `raw-sim` instead of duplicating simulation logic inside JDD. Updates to lens PSF, CFA packing, or noise modeling in `raw-sim` automatically flow into JDD training.

## Model

The current network is a compact NAFNet-style restoration model:

```text
packed RAW burst + noise map
  -> input convolution
  -> NAF blocks
  -> output convolution
  -> PixelShuffle CFA upsampling
  -> clean linear RGB
```

The PixelShuffle scale follows the CFA type:

| CFA type | Packed scale | RAW channels per frame | Input channels for 3 frames |
| --- | ---: | ---: | ---: |
| Bayer / binning | 2 | 4 | 13 |
| Quad Bayer | 4 | 16 | 49 |

## Main Training Command

```bash
cd JDD
bash ./scripts/train.sh --config ./configs/train_JDD_patch128_3f_iter100000.json
```

Useful training fields:

| Field | Purpose |
| --- | --- |
| `data.simulate_on_device` | Use GPU-side `raw-sim` batch degradation. |
| `data.frames` | Number of burst frames. Current JDD expects 3. |
| `data.noise_map_reduce` | Reduces per-channel noise maps before appending the map channel. |
| `train.batch_size` | Number of RGB patches per step. |
| `train.amp` | Mixed-precision training on CUDA. |
| `train.channels_last` | Channels-last memory format for CUDA kernels. |
| `train.val_every` | Validation interval. |

