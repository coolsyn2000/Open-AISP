from __future__ import annotations

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target).square() + self.eps * self.eps).mean()
