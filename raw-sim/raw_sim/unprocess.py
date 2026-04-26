from __future__ import annotations

from typing import Any

import torch

from .images import normalize_srgb


def inverse_gamma(srgb: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return srgb.clamp(0.0, 1.0).pow(float(gamma)).clamp(0.0, 1.0)


def linear_rgb_gt_from_srgb(srgb_image: torch.Tensor, camera: dict[str, Any]) -> torch.Tensor:
    return inverse_gamma(normalize_srgb(srgb_image), float(camera["gamma"]))


def srgb_to_camera_rgb(srgb_image: torch.Tensor, camera: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    linear_rgb = linear_rgb_gt_from_srgb(srgb_image, camera)
    ccm = torch.tensor(camera["ccm"]["matrix"], dtype=linear_rgb.dtype, device=linear_rgb.device)
    if ccm.shape != (3, 3):
        raise ValueError("camera.ccm.matrix must be 3x3")
    inv_ccm = torch.linalg.inv(ccm)
    camera_rgb = linear_rgb @ inv_ccm.T
    awb = camera["awb"]
    gains = torch.tensor(
        [float(awb["red_gain"]), float(awb.get("green_gain", 1.0)), float(awb["blue_gain"])],
        dtype=camera_rgb.dtype,
        device=camera_rgb.device,
    )
    camera_rgb = (camera_rgb / gains.view(1, 1, 3).clamp_min(1e-8)).clamp(0.0, 1.0)
    return camera_rgb, {
        "gamma": float(camera["gamma"]),
        "ccm": ccm.detach().cpu().tolist(),
        "awb_gains": {
            "red_gain": float(awb["red_gain"]),
            "green_gain": float(awb.get("green_gain", 1.0)),
            "blue_gain": float(awb["blue_gain"]),
        },
    }

