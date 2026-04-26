from pathlib import Path

import numpy as np
import torch
from PIL import Image

from raw_sim.config import load_camera_json
from raw_sim.pipeline import simulate_image


def test_camera_json_pipeline_outputs_training_pair(tmp_path: Path):
    image_path = tmp_path / "input.png"
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8)).save(image_path)
    camera = load_camera_json(Path(__file__).parents[1] / "configs" / "cameras" / "example_camera_10bit_binning.json")
    result = simulate_image(image_path, camera, patch_size=16, seed=12)
    meta = result["metadata"]
    assert result["input_noisy_raw"].shape == (8, 8, 4)
    assert result["gt_linear_rgb"].shape == (16, 16, 3)
    assert meta["camera"]["analog_gain"] == camera["camera"]["analog_gain"]
    assert meta["camera"]["iso"] == int(round(camera["camera"]["analog_gain"] * 100.0))
    assert meta["sensor"]["black_level"] == 32
    assert meta["shapes"]["input_noisy_raw"] == [8, 8, 4]
    assert meta["shapes"]["gt_linear_rgb"] == [16, 16, 3]
    assert meta["raw_visualization_shape"] == [16, 16]


def test_lens_psf_affects_raw_but_not_linear_rgb_gt(tmp_path: Path):
    image_path = tmp_path / "edge.png"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:, :] = 255
    Image.fromarray(image).save(image_path)

    camera = load_camera_json(Path(__file__).parents[1] / "configs" / "cameras" / "example_camera_10bit_binning.json")
    camera["noise"]["strength_multiplier"] = 0.0

    sharp_camera = dict(camera)
    sharp_camera["lens_psf"] = {"enabled": False, "kernel_size": 5, "sigma": 1.0}
    blurred_camera = dict(camera)
    blurred_camera["lens_psf"] = {"enabled": True, "kernel_size": 5, "sigma": 1.0}

    sharp = simulate_image(image_path, sharp_camera, patch_size=32, seed=3)
    blurred = simulate_image(image_path, blurred_camera, patch_size=32, seed=3)

    assert torch.allclose(sharp["gt_linear_rgb"], blurred["gt_linear_rgb"])
    assert not torch.allclose(sharp["input_noisy_raw"], blurred["input_noisy_raw"])
