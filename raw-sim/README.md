# raw-sim

`raw-sim` converts high-quality RGB images into training pairs for RAW denoise/demosaic models:

- `sample_input_noisy_raw.raw`: noisy packed RAW input, `uint16`.
- `sample_gt_linear_rgb.raw`: high-quality linear RGB GT, `float32`, HWC RGB in `[0, 1]`.
- `sample_iso.txt`: ISO value for the sample.
- `sample_metadata.json`: shapes, dtypes, CFA packing, camera parameters and noise parameters.

Shape convention:

- Bayer full readout: RGB `H x W x 3` -> RAW `H/2 x W/2 x 4`
- Quad Bayer filter: RGB `H x W x 3` -> RAW `H/4 x W/4 x 16`
- Binning filter: RGB `H x W x 3` -> RAW `H/2 x W/2 x 4`

`binning_operation: "average"` means a local bin uses the average value rather than the sum, so the signal scale remains comparable to full readout. The PNG visualization always unpacks RAW back to the original spatial size for inspection.

The code is intentionally flat and small:

```text
scripts/
  generate-raw.sh
  download-hr-datasets.sh
  test.sh
raw_sim/
  cli.py          # called by scripts/generate-raw.sh
  download.py     # called by scripts/download-hr-datasets.sh
  pipeline.py     # RGB -> noisy RAW + linear RGB GT
  unprocess.py
  cfa.py
  noise.py
  sensor.py
  outputs.py
  images.py
  config.py
configs/
  cameras/
    example_camera_10bit_quadbayer.json
    example_camera_10bit_binning.json
```

## Camera JSON Only

Simulation uses one camera JSON file. It contains gamma, CCM, AWB, lens PSF blur, sensor bit depth and levels, CFA/readout mode, analog gain, and calibrated noise parameters.

ISO is defined as:

```text
ISO = analog_gain * 100
```

In camera JSON, `camera.analog_gain` is the only input control for noise level. The simulator derives ISO as `analog_gain * 100` for metadata and `sample_iso.txt`, then scales the calibrated Poisson-Gaussian noise parameters from analog gain.

Black level can be `"auto"`. With `black_level_10bit_reference: 32`, auto black level is:

- 10-bit: 32 DN
- 12-bit: 128 DN
- 14-bit: 512 DN

Lens PSF blur is configured under `lens_psf`, and it is applied before sensor level mapping and CFA packing. Supported kernels:

- `kernel: "gaussian"` with `kernel_size` and `sigma`.
- `kernel: "guided"` with `kernel_size` and `eps` for edge-aware guided filtering.

## Download RGB Datasets

DIV2K HR:

```bash
python ./scripts/download_hr_datasets.py --dataset div2k --root ./datasets --div2k-splits train
```

Full Flickr2K HR dataset from Hugging Face (about 11.6 GB, 2650 images):

```bash
python ./scripts/download_hr_datasets.py --dataset flickr2k --root ./datasets
```

## Generate Training Pairs

Run through the bash entrypoint:

```bash
python ./scripts/generate_raw.py \
  --input ./datasets/DIV2K/DIV2K_train_HR \
  --output ./simu_pairs \
  --camera-json ./configs/cameras/example_camera_10bit_RGGB_binning.json \
  --patch-size 512 \
  --num-patches 5
```
