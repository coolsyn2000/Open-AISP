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


def _box_filter(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    x = F.pad(x, (pad_left, pad_right, pad_left, pad_right), mode="reflect")
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1)


def _guided_filter_nchw(x: torch.Tensor, kernel_size: int, eps: float) -> torch.Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("lens_psf.kernel_size must be a positive odd integer")
    if eps <= 0:
        raise ValueError("lens_psf.eps must be positive")

    guide = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0.0, 1.0)
    mean_guide = _box_filter(guide, kernel_size)
    corr_guide = _box_filter(guide * guide, kernel_size)
    var_guide = corr_guide - mean_guide.square()

    outputs = []
    for channel in range(x.shape[1]):
        src = x[:, channel : channel + 1]
        mean_src = _box_filter(src, kernel_size)
        corr_guide_src = _box_filter(guide * src, kernel_size)
        cov_guide_src = corr_guide_src - mean_guide * mean_src
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        mean_a = _box_filter(a, kernel_size)
        mean_b = _box_filter(b, kernel_size)
        outputs.append(mean_a * guide + mean_b)
    return torch.cat(outputs, dim=1).clamp(0.0, 1.0)


def apply_lens_psf_batch(camera_rgb: torch.Tensor, camera: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    psf = camera.get("lens_psf", {})
    enabled = bool(psf.get("enabled", True))
    kernel = str(psf.get("kernel", "gaussian")).lower()
    kernel_size = int(psf.get("kernel_size", 5))
    sigma = float(psf.get("sigma", 1.0))
    eps = float(psf.get("eps", 1e-3))
    if not enabled:
        return camera_rgb, {"enabled": False, "kernel": kernel, "kernel_size": kernel_size, "sigma": sigma, "eps": eps}

    if kernel == "gaussian":
        weight_kernel = _gaussian_kernel(kernel_size, sigma, camera_rgb.dtype, camera_rgb.device)
        weight = weight_kernel.view(1, 1, kernel_size, kernel_size).repeat(camera_rgb.shape[1], 1, 1, 1)
        pad = kernel_size // 2
        x = F.pad(camera_rgb, (pad, pad, pad, pad), mode="reflect")
        out = F.conv2d(x, weight, groups=camera_rgb.shape[1])
    elif kernel in {"guided", "guided_filter"}:
        out = _guided_filter_nchw(camera_rgb, kernel_size, eps)
        kernel = "guided"
    else:
        raise ValueError(f"Unsupported lens_psf.kernel '{kernel}'. Use 'gaussian' or 'guided'.")
    return out.clamp(0.0, 1.0), {"enabled": True, "kernel": kernel, "kernel_size": kernel_size, "sigma": sigma, "eps": eps}


def apply_lens_psf(camera_rgb: torch.Tensor, camera: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    psf = camera.get("lens_psf", {})
    x = camera_rgb.permute(2, 0, 1).unsqueeze(0)
    y, meta = apply_lens_psf_batch(x, camera)
    out = y.squeeze(0).permute(1, 2, 0).contiguous()
    return out.clamp(0.0, 1.0), meta
