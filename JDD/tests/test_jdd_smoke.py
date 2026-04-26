from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jdd.data import SimulatedRawBurstDataset
from jdd.infer import opencv_demosaic_baseline
from jdd.model import JointDenoiseDemosaicNAFNet
from jdd.train import resolve_config_paths
from jdd.utils import load_json
from raw_sim.config import load_camera_json


def test_dataset_makes_three_frame_raw_burst(tmp_path: Path):
    image_path = tmp_path / "rgb.png"
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(image_path)
    camera_module_json = Path(__file__).resolve().parents[1] / "configs" / "camera_module_10bit_binning_precali_noise_rggb_ag1to64.json"
    dataset = SimulatedRawBurstDataset(image_path, camera_module_json, patch_size=64, length=1, seed=3)
    sample = dataset[0]
    assert sample["input"].shape == (13, 32, 32)
    assert sample["target"].shape == (3, 64, 64)
    assert 100 <= int(sample["iso"].item()) <= 6400


def test_model_outputs_full_resolution_linear_rgb():
    model = JointDenoiseDemosaicNAFNet(in_channels=13, cfa_scale=2, width=8, middle_blocks=1)
    y = model(torch.rand(1, 13, 16, 16))
    assert y.shape == (1, 3, 32, 32)


def test_train_config_loads_separate_camera_module_json(tmp_path: Path):
    for i in range(3):
        image_path = tmp_path / f"rgb_{i}.png"
        Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(image_path)

    config_path = Path(__file__).resolve().parents[1] / "configs" / "train_JDD_patch128_3f_iter100000.json"
    config = resolve_config_paths(load_json(config_path), config_path)
    config["data"]["image_roots"] = [str(tmp_path)]
    config["data"]["max_images"] = 2
    config["data"]["patch_size"] = 64

    dataset = SimulatedRawBurstDataset(
        image_roots=config["data"]["image_roots"],
        camera_module_json=config["camera_module_json"],
        patch_size=config["data"]["patch_size"],
        max_images=config["data"]["max_images"],
        seed=11,
    )
    assert len(dataset) == 2
    iso_values = [int(dataset[i]["iso"].item()) for i in range(len(dataset))]
    assert min(iso_values) >= 100
    assert max(iso_values) <= 6400


def test_dataset_does_not_expand_one_image_to_many_samples(tmp_path: Path):
    image_path = tmp_path / "single.png"
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(image_path)
    camera_module_json = Path(__file__).resolve().parents[1] / "configs" / "camera_module_10bit_binning_precali_noise_rggb_ag1to64.json"
    dataset = SimulatedRawBurstDataset(tmp_path, camera_module_json, patch_size=64, length=128, seed=3)
    assert len(dataset) == 1


def test_opencv_demosaic_baseline_outputs_linear_rgb(tmp_path: Path):
    image_path = tmp_path / "rgb.png"
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(image_path)
    camera_module_json = Path(__file__).resolve().parents[1] / "configs" / "camera_module_10bit_binning_precali_noise_rggb_ag1to64.json"
    camera = load_camera_json(camera_module_json)
    dataset = SimulatedRawBurstDataset(image_path, camera_module_json, patch_size=64, length=1, seed=3)
    sample = dataset[0]
    baseline = opencv_demosaic_baseline(sample["input"].unsqueeze(0), camera, raw_channels=4, scale=2)
    assert baseline.shape == (1, 3, 64, 64)
    assert torch.all((0.0 <= baseline) & (baseline <= 1.0))


def test_opencv_demosaic_rggb_color_order_is_rgb():
    camera = {
        "sensor": {"bit_depth": 10, "black_level": 0, "white_level": 1023},
        "cfa": {"pattern": "RGGB"},
        "awb": {"red_gain": 1.0, "green_gain": 1.0, "blue_gain": 1.0},
        "ccm": {"matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
    }
    x = torch.zeros(1, 13, 8, 8)
    x[:, 0] = 1.0
    x[:, 1] = 0.5
    x[:, 2] = 0.5
    x[:, 3] = 0.2
    baseline = opencv_demosaic_baseline(x, camera, raw_channels=4, scale=2)
    mean_rgb = baseline[:, :, 4:-4, 4:-4].mean(dim=(0, 2, 3))
    assert mean_rgb[0] > mean_rgb[1] > mean_rgb[2]
