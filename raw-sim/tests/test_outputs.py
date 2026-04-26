import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from raw_sim.outputs import save_training_pair


def test_save_training_pair_files(tmp_path: Path):
    metadata = {
        "sensor": {"bit_depth": 10},
        "camera": {"iso": 800},
        "cfa": {"type": "bayer", "pattern": "RGGB"},
        "readout": {"mode": "full"},
        "shapes": {"gt_linear_rgb": [4, 4, 3]},
    }
    noisy = torch.rand(2, 2, 4)
    gt = torch.rand(4, 4, 3)
    save_training_pair(tmp_path, noisy, gt, metadata, stem="sample")

    assert (tmp_path / "sample_input_noisy_raw.raw").exists()
    assert (tmp_path / "sample_gt_linear_rgb.raw").exists()
    assert (tmp_path / "sample_input_noisy_raw.png").exists()
    assert (tmp_path / "sample_gt_linear_rgb.png").exists()
    assert (tmp_path / "sample_iso.txt").read_text(encoding="utf-8") == "800"
    saved = json.loads((tmp_path / "sample_metadata.json").read_text(encoding="utf-8"))
    assert saved["files"]["input_noisy_raw"]["shape"] == [2, 2, 4]
    assert saved["files"]["gt_linear_rgb"]["shape"] == [4, 4, 3]
    assert np.fromfile(tmp_path / "sample_input_noisy_raw.raw", dtype="<u2").max() <= 1023
    assert np.fromfile(tmp_path / "sample_gt_linear_rgb.raw", dtype="<f4").shape[0] == 4 * 4 * 3
    with Image.open(tmp_path / "sample_input_noisy_raw.png") as image:
        assert image.mode == "L"
        assert image.size == (4, 4)
    with Image.open(tmp_path / "sample_gt_linear_rgb.png") as image:
        assert image.mode == "RGB"
        assert image.size == (4, 4)
