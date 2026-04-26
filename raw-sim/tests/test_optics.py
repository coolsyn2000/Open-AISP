import torch

from raw_sim.optics import apply_lens_psf


def test_lens_psf_keeps_shape_and_spreads_impulse():
    x = torch.zeros(9, 9, 3)
    x[4, 4, :] = 1.0
    y, meta = apply_lens_psf(x, {"lens_psf": {"enabled": True, "kernel_size": 5, "sigma": 1.0}})
    assert y.shape == x.shape
    assert meta["kernel_size"] == 5
    assert torch.all(y[4, 4] < 1.0)
    assert torch.all(y[4, 5] > 0.0)
