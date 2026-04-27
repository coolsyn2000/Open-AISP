# JDD: Joint Denoise and Demosaic

This folder trains a CNN for supervised AI-ISP from simulated RGB data:

```text
RGB image -> raw-sim degradation -> 3 noisy packed RAW frames + noise map -> JDD model -> clean linear RGB
```

The simulation is not duplicated here. `JDD` imports the existing `raw-sim` code for inverse gamma, CCM/AWB, lens PSF blur, sensor levels, CFA packing, analog-gain-dependent Poisson-Gaussian noise, noise-map estimation, and CUDA batch degradation.

## Input and Target

For one training sample:

- Input: 3 noisy RAW frames with the same analog gain plus 1 grayscale noise map.
- Frame order: frame 0 is the base frame, frames 1-2 are supplemental noisy frames sampled from the same clean RAW and ISO.
- Bayer/binning shape: `(3 * 4 + 1, H/2, W/2)`.
- Quad Bayer shape: `(3 * 16 + 1, H/4, W/4)`.
- Target: clean linear RGB, shape `(3, H, W)`.
- Analog gain: sampled from the camera module JSON, for example `[1, 64]`.
- ISO: derived for logging and metadata with `ISO = analog_gain * 100`.

The network uses a compact NAFNet-style backbone at packed RAW resolution and PixelShuffle upsamples to linear RGB.

## Quick Start

```
cd ./JDD
bash ./scripts/train.sh --config ./configs/train_JDD_patch128_3f_iter100000.json
```

Training uses tqdm for progress and writes the same key events to `train.output_dir/train.log`. Checkpoints are saved as `.pth`, including `latest.pth` and `step_XXXXXXXX.pth`.

Each run loads two JSON files:

- `configs/camera_module_10bit_binning_precali_noise_rggb_ag1to64.json`: camera module and degradation parameters. This follows the `raw-sim` camera format, except `camera.analog_gain` is a range.
- `configs/train_JDD_patch128_3f_iter100000.json`: training parameters such as RGB data path, patch size, loss, iterations, batch size, learning rate, and checkpoint path.


- `data.image_roots`: list of RGB image folders, so one run can mix DIV2K, Flickr2K, and other datasets.
- `camera_module_json`: camera module JSON used for RAW degradation.
- `data.patch_size`: must be divisible by the CFA scale, 2 for Bayer/binning and 4 for Quad Bayer.
- Training samples one random crop/noise realization from each RGB image whenever that image is visited; `train.iterations` controls total update count.
- `data.max_images`: optional cap on the number of training RGB images. It does not repeat images to create more samples.
- `data.cache_images`, `data.max_cached_images`: cache decoded RGB images inside persistent workers. This is useful for PNG-heavy DIV2K/Flickr2K training.
- `data.simulate_on_device`: when `true` on CUDA, workers only load/crop RGB patches and RAW degradation is generated as a vectorized GPU batch. This avoids CPU online simulation becoming the bottleneck.
- `data.val.image_roots`, `data.val.max_images`: small validation RGB set, usually a few dozen images.
- `train.iterations`: total optimizer update count.
- `device`: use `"cuda"` for GPU training.
- `gpu_id`: select the GPU index, for example `0` or `1`.
- `train.num_workers`, `train.persistent_workers`, `train.prefetch_factor`: keep online RAW simulation workers alive and prefetch batches. This is important on Windows because worker startup is expensive.
- `train.worker_torch_threads`: set to `1` so each data worker does not spawn many CPU torch threads and fight other workers.
- `train.shuffle_images`: default `false` for speed with per-worker image cache. Random crop/noise/analog gain still change every visit.
- `train.amp`: mixed precision training on CUDA.
- `train.channels_last`: use CUDA channels-last tensors for convolution throughput.
- `train.compile`: optionally enable `torch.compile` for the model. Keep it disabled if startup time or checkpoint compatibility is more important.

The first batch can still be slow because workers start and decode/cache large PNG images. In the optimized path, `train.log` reports `last_data_time` and `last_iter_time`; after warmup, `last_data_time` should be close to zero if the GPU is no longer waiting for the data loader.

Edit the camera module JSON for degradation changes:

- `camera.analog_gain.min`, `camera.analog_gain.max`: sampled analog gain range.
- `camera.analog_gain_sampling`: `log_uniform` or `uniform`.
- `noise.calibration`: calibrated per-channel Poisson-Gaussian noise parameters.

Checkpoints are saved to `train.output_dir`.

## Inference

Run full-image inference with pretrained weights:

```bash
python ./scripts/infer.py \
  --checkpoint ./assets/latest.pth \
  --input ../raw-sim/datasets/Flickr2K/Flickr2K_HR \
  --output ./runs/example_binning/infer_full \
  --camera-module-json ./configs/camera_module_10bit_binning_precali_noise_rggb_ag1to64.json \
  --mode full \
  --max-images 1
```

Each image output folder contains `pred.png`, `opencv_demosaic.png`, `gt.png`, `comparison_gt_pred_error.png`, `comparison_gt_jdd_opencv_errors.png`, `metrics.json`, and metadata. The OpenCV image is a direct demosaic baseline from the noisy base RAW frame, followed by black-level removal, AWB, and CCM back to linear RGB. The output root also contains `metrics.csv` with JDD and OpenCV PSNR, SSIM, and LPIPS. 
