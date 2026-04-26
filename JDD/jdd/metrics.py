from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    mse = (pred.clamp(0, 1) - target.clamp(0, 1)).square().flatten(1).mean(dim=1)
    return (-10.0 * torch.log10(mse.clamp_min(eps))).mean()


def _ssim_window(channels: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    coords = torch.arange(11, dtype=dtype, device=device) - 5
    kernel_1d = torch.exp(-(coords.square()) / (2 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.view(1, 1, 11, 11).repeat(channels, 1, 1, 1)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    channels = pred.shape[1]
    window = _ssim_window(channels, pred.dtype, pred.device)
    mu_x = F.conv2d(pred, window, padding=5, groups=channels)
    mu_y = F.conv2d(target, window, padding=5, groups=channels)
    sigma_x = F.conv2d(pred * pred, window, padding=5, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(target * target, window, padding=5, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(pred * target, window, padding=5, groups=channels) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))
    return score.flatten(1).mean(dim=1).mean()


@dataclass
class MetricResult:
    psnr: float
    ssim: float
    lpips: float


class MetricComputer:
    def __init__(self, device: torch.device, use_lpips: bool = True) -> None:
        self.lpips_model = None
        if use_lpips:
            try:
                import lpips  # type: ignore

                self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()
                for param in self.lpips_model.parameters():
                    param.requires_grad_(False)
            except Exception:
                self.lpips_model = None

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> MetricResult:
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)
        lpips_value = float("nan")
        if self.lpips_model is not None:
            lpips_value = float(self.lpips_model(pred * 2 - 1, target * 2 - 1).mean().item())
        return MetricResult(psnr=float(psnr(pred, target).item()), ssim=float(ssim(pred, target).item()), lpips=lpips_value)
