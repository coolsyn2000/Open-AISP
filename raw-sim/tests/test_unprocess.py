import torch

from raw_sim.unprocess import inverse_gamma, linear_rgb_gt_from_srgb, srgb_to_camera_rgb


CAMERA = {
    "gamma": 2.2,
    "awb": {"red_gain": 2.0, "green_gain": 1.0, "blue_gain": 4.0},
    "ccm": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
}


def test_linear_rgb_gt_range():
    x = torch.linspace(0, 1, 16).view(4, 4, 1).repeat(1, 1, 3)
    y = inverse_gamma(x, 2.2)
    assert y.min() >= 0
    assert y.max() <= 1


def test_gt_shape_and_camera_awb():
    x = torch.ones(2, 2, 3)
    gt = linear_rgb_gt_from_srgb(x, CAMERA)
    camera_rgb, _ = srgb_to_camera_rgb(x, CAMERA)
    assert gt.shape == x.shape
    assert torch.allclose(camera_rgb[..., 0], torch.full((2, 2), 0.5))
    assert torch.allclose(camera_rgb[..., 1], torch.ones(2, 2))
    assert torch.allclose(camera_rgb[..., 2], torch.full((2, 2), 0.25))

