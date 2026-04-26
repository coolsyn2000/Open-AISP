from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _gaussian_kernel(kernel_size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("lens_psf.kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("lens_psf.sigma must be positive")
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * float(sigma) ** 2))
    return kernel / kernel.sum().clamp_min(1e-12)


def apply_lens_psf(camera_rgb: torch.Tensor, camera: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    psf = camera.get("lens_psf", {})
    enabled = bool(psf.get("enabled", True))
    kernel_size = int(psf.get("kernel_size", 5))
    sigma = float(psf.get("sigma", 1.0))
    if not enabled:
        return camera_rgb, {"enabled": False, "kernel": "gaussian", "kernel_size": kernel_size, "sigma": sigma}

    kernel = _gaussian_kernel(kernel_size, sigma, camera_rgb.dtype, camera_rgb.device)
    weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    x = camera_rgb.permute(2, 0, 1).unsqueeze(0)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.conv2d(x, weight, groups=3)
    out = y.squeeze(0).permute(1, 2, 0).contiguous()
    return out.clamp(0.0, 1.0), {"enabled": True, "kernel": "gaussian", "kernel_size": kernel_size, "sigma": sigma}
