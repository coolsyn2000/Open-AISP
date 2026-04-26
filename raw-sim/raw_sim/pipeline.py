from __future__ import annotations

from pathlib import Path
from typing import Any

from .cfa import make_raw
from .images import load_rgb_image, random_patch
from .noise import iso_value, noise_params, simulate_noise, analog_gain
from .optics import apply_lens_psf
from .sensor import apply_sensor_levels, sensor_level_dn
from .unprocess import linear_rgb_gt_from_srgb, srgb_to_camera_rgb


def simulate_image(image_path: str | Path, camera: dict[str, Any], patch_size: int | None = None, seed: int | None = None) -> dict[str, Any]:
    image_path = Path(image_path)
    image = load_rgb_image(image_path)
    patch, crop = random_patch(image, patch_size, seed)

    linear_rgb_gt = linear_rgb_gt_from_srgb(patch, camera)
    camera_rgb, unprocess_meta = srgb_to_camera_rgb(patch, camera)
    camera_rgb, psf_meta = apply_lens_psf(camera_rgb, camera)
    sensor_rgb = apply_sensor_levels(camera_rgb, camera["sensor"])
    clean_raw, channel_names, packing_order, samples_per_channel = make_raw(sensor_rgb, camera)
    params = noise_params(camera, channel_names)
    noisy_raw = simulate_noise(clean_raw, params, channel_names, camera, samples_per_channel, seed)

    bit_depth = int(camera["sensor"]["bit_depth"])
    metadata = {
        "camera": {
            "name": camera.get("camera", {}).get("name", "unknown"),
            "analog_gain": analog_gain(camera),
            "iso": iso_value(camera),
            "iso_definition": "ISO = analog_gain * 100",
        },
        "source_image_path": str(image_path.resolve()),
        "crop_position": crop,
        "random_seed": seed,
        "gamma": unprocess_meta["gamma"],
        "ccm": unprocess_meta["ccm"],
        "awb_gains": unprocess_meta["awb_gains"],
        "sensor": {
            "bit_depth": bit_depth,
            "black_level": sensor_level_dn(camera["sensor"], "black_level"),
            "white_level": sensor_level_dn(camera["sensor"], "white_level"),
            "full_well": float(camera["sensor"].get("full_well", 10000.0)),
        },
        "cfa": {
            "type": camera["cfa"].get("type", "bayer"),
            "pattern": camera["cfa"].get("pattern", "RGGB").upper(),
            "packing_order": packing_order,
        },
        "readout": dict(camera.get("readout", {"mode": "full"})),
        "lens_psf": psf_meta,
        "noise": {
            "model": camera["noise"].get("model", "poisson_gaussian"),
            "channel_names": channel_names,
            "channel_params": params,
            "samples_per_channel": samples_per_channel,
        },
        "shapes": {
            "input_noisy_raw": list(noisy_raw.shape),
            "gt_linear_rgb": list(linear_rgb_gt.shape),
        },
        "raw_visualization_shape": list(linear_rgb_gt.shape[:2]),
    }
    return {"input_noisy_raw": noisy_raw, "gt_linear_rgb": linear_rgb_gt, "metadata": metadata}
