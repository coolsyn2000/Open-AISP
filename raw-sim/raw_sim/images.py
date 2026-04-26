from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def collect_images(root: str | Path) -> list[Path]:
    path = Path(root)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_rgb_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"))
    return torch.from_numpy(arr.copy())


def normalize_srgb(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got {tuple(image.shape)}")
    if torch.is_floating_point(image):
        x = image.to(torch.float32)
        if x.max() > 1.5:
            x = x / 255.0
    else:
        x = image.to(torch.float32) / (65535.0 if image.max() > 255 else 255.0)
    if x.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {tuple(image.shape)}")
    return x.clamp(0.0, 1.0)


def random_patch(image: torch.Tensor, patch_size: int | None, seed: int | None = None) -> tuple[torch.Tensor, dict[str, int]]:
    if patch_size is None or patch_size <= 0:
        return image, {"top": 0, "left": 0, "height": int(image.shape[0]), "width": int(image.shape[1])}
    h, w = int(image.shape[0]), int(image.shape[1])
    if h < patch_size or w < patch_size:
        scale = patch_size / min(h, w)
        resized = Image.fromarray(image.cpu().numpy()).resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BICUBIC)
        image = torch.from_numpy(np.asarray(resized).copy())
        h, w = int(image.shape[0]), int(image.shape[1])
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    top = int(torch.randint(0, h - patch_size + 1, (), generator=gen).item()) if h > patch_size else 0
    left = int(torch.randint(0, w - patch_size + 1, (), generator=gen).item()) if w > patch_size else 0
    patch = image[top : top + patch_size, left : left + patch_size, :]
    return patch, {"top": top, "left": left, "height": int(patch.shape[0]), "width": int(patch.shape[1])}

