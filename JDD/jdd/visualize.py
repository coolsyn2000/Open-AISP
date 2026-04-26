from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_rgb_image(x: torch.Tensor) -> Image.Image:
    if x.ndim == 4:
        x = x[0]
    arr = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = np.round(arr ** (1.0 / 2.2) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_rgb_tensor(path: str | Path, x: torch.Tensor) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_rgb_image(x).save(out)
