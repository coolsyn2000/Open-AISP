from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .sensor import max_dn


def _uint_raw(x: torch.Tensor, bit_depth: int) -> np.ndarray:
    levels = max_dn(bit_depth)
    arr = np.round(x.detach().cpu().clamp(0, 1).numpy() * levels).astype(np.uint16)
    return np.clip(arr, 0, levels).astype(np.uint16)


def _float_raw(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().clamp(0, 1).numpy().astype("<f4")


def _unpack_for_png(raw: np.ndarray, metadata: dict) -> np.ndarray:
    if raw.ndim != 3:
        return raw
    h, w, channels = raw.shape
    if channels == 1:
        return raw[..., 0]
    if channels == 4:
        mosaic = np.empty((h * 2, w * 2), dtype=raw.dtype)
        mosaic[0::2, 0::2] = raw[..., 0]
        mosaic[0::2, 1::2] = raw[..., 1]
        mosaic[1::2, 0::2] = raw[..., 2]
        mosaic[1::2, 1::2] = raw[..., 3]
        return mosaic
    if channels == 16:
        mosaic = np.empty((h * 4, w * 4), dtype=raw.dtype)
        ch = 0
        for r in range(4):
            for c in range(4):
                mosaic[r::4, c::4] = raw[..., ch]
                ch += 1
        return mosaic
    raise ValueError(f"Cannot visualize RAW shape {raw.shape}")


def _save_raw_png(path: Path, raw: np.ndarray, metadata: dict) -> None:
    mosaic = _unpack_for_png(raw, metadata)
    bit_depth = int(metadata["sensor"]["bit_depth"])
    image = np.round(mosaic.astype(np.float32) / max_dn(bit_depth) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def _save_linear_rgb_png(path: Path, linear_rgb: torch.Tensor) -> None:
    arr = linear_rgb.detach().cpu().clamp(0, 1).numpy()
    image = np.round(arr ** (1.0 / 2.2) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def save_training_pair(output_dir: str | Path, noisy_raw: torch.Tensor, linear_rgb_gt: torch.Tensor, metadata: dict, stem: str = "sample") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bit_depth = int(metadata["sensor"]["bit_depth"])

    raw_path = out / f"{stem}_input_noisy_raw.raw"
    raw_arr = _uint_raw(noisy_raw, bit_depth)
    raw_arr.tofile(raw_path)

    gt_path = out / f"{stem}_gt_linear_rgb.raw"
    gt_arr = _float_raw(linear_rgb_gt)
    gt_arr.tofile(gt_path)

    raw_png_path = out / f"{stem}_input_noisy_raw.png"
    gt_png_path = out / f"{stem}_gt_linear_rgb.png"
    _save_raw_png(raw_png_path, raw_arr, metadata)
    _save_linear_rgb_png(gt_png_path, linear_rgb_gt)

    iso_path = out / f"{stem}_iso.txt"
    iso_path.write_text(str(metadata["camera"]["iso"]), encoding="utf-8")

    metadata["files"] = {
        "input_noisy_raw": {
            "path": str(raw_path.resolve()),
            "shape": list(raw_arr.shape),
            "dtype": "uint16",
            "byte_order": "little_endian",
            "layout": "packed_hwc_interleaved",
            "bit_depth": bit_depth,
            "valid_dn_range": [0, max_dn(bit_depth)],
        },
        "gt_linear_rgb": {
            "path": str(gt_path.resolve()),
            "shape": list(gt_arr.shape),
            "dtype": "float32",
            "byte_order": "little_endian",
            "layout": "hwc_rgb",
            "range": [0.0, 1.0],
        },
        "iso": {"path": str(iso_path.resolve()), "value": metadata["camera"]["iso"]},
        "input_noisy_raw_png": {
            "path": str(raw_png_path.resolve()),
            "layout": "mosaic_grayscale",
            "note": "packed RAW restored to original CFA spatial arrangement for visualization",
        },
        "gt_linear_rgb_png": {
            "path": str(gt_png_path.resolve()),
            "layout": "rgb_preview",
            "note": "linear RGB visualized with display gamma 1/2.2",
        },
    }
    with (out / f"{stem}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
