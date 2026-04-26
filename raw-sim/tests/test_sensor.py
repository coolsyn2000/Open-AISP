import torch

from raw_sim.sensor import apply_sensor_levels, default_black_level_dn


def test_auto_black_level_scales_with_bit_depth():
    assert default_black_level_dn(10) == 32
    assert default_black_level_dn(12) == 128
    assert default_black_level_dn(14) == 512
    sensor = {"bit_depth": 12, "black_level": "auto", "white_level": "auto", "quantize": True}
    dark = apply_sensor_levels(torch.zeros(2, 2, 3), sensor)
    assert torch.equal(torch.round(dark * 4095), torch.full((2, 2, 3), 128.0))

