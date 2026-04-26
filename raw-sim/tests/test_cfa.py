import torch

from raw_sim.cfa import make_raw, quad_tile, tile_colors


def test_bayer_full_is_4ch_half_resolution():
    camera = {"cfa": {"type": "bayer", "pattern": "RGGB"}, "readout": {"mode": "full"}}
    raw, names, order, samples = make_raw(torch.rand(10, 12, 3), camera)
    assert raw.shape == (5, 6, 4)
    assert names == ["R", "Gr", "Gb", "B"]
    assert order == names
    assert samples == 1


def test_bayer_packing_uses_rggb_samples():
    rgb = torch.zeros(4, 4, 3)
    rgb[..., 0] = 0.2
    rgb[..., 1] = 0.5
    rgb[..., 2] = 0.8
    camera = {"cfa": {"type": "bayer", "pattern": "RGGB"}, "readout": {"mode": "full"}}
    raw, _, _, _ = make_raw(rgb, camera)
    assert torch.allclose(raw[0, 0], torch.tensor([0.2, 0.5, 0.5, 0.8]))


def test_binning_is_4ch_half_resolution():
    camera = {
        "cfa": {"type": "binning", "pattern": "RGGB"},
        "readout": {"mode": "full", "binning_type": "analog_binning", "binning_operation": "average"},
    }
    raw, names, _, samples = make_raw(torch.rand(8, 8, 3), camera)
    assert raw.shape == (4, 4, 4)
    assert names == ["R", "Gr", "Gb", "B"]
    assert samples == 4


def test_quad_bayer_full_is_16ch_quarter_resolution():
    camera = {"cfa": {"type": "quadbayer", "pattern": "RGGB"}, "readout": {"mode": "full"}}
    raw, names, order, samples = make_raw(torch.rand(8, 8, 3), camera)
    assert raw.shape == (2, 2, 16)
    assert names == [f"ch{i}" for i in range(16)]
    assert order[0]["color"] == "R"
    assert order[-1]["color"] == "B"
    assert samples == 1


def test_quad_bayer_packing_uses_4x4_tile():
    rgb = torch.zeros(4, 4, 3)
    rgb[..., 0] = 0.2
    rgb[..., 1] = 0.5
    rgb[..., 2] = 0.8
    camera = {"cfa": {"type": "quadbayer", "pattern": "RGGB"}, "readout": {"mode": "full"}}
    raw, _, _, _ = make_raw(rgb, camera)
    assert torch.allclose(raw[0, 0], torch.tensor([0.2, 0.2, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.5, 0.5, 0.8, 0.8]))
