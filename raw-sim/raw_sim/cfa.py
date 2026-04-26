from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

BAYER_PATTERNS = {
    "RGGB": (("R", "Gr"), ("Gb", "B")),
    "BGGR": (("B", "Gb"), ("Gr", "R")),
    "GRBG": (("Gr", "R"), ("B", "Gb")),
    "GBRG": (("Gb", "B"), ("R", "Gr")),
}
COLOR_INDEX = {"R": 0, "Gr": 1, "Gb": 1, "G": 1, "B": 2}


def bayer_tile(pattern: str) -> list[list[str]]:
    return [list(row) for row in BAYER_PATTERNS[pattern.upper()]]


def bayer_channel_names(pattern: str) -> list[str]:
    tile = bayer_tile(pattern)
    return [tile[0][0], tile[0][1], tile[1][0], tile[1][1]]


def quad_tile(pattern: str) -> list[list[str]]:
    base = BAYER_PATTERNS[pattern.upper()]
    tile = [["" for _ in range(4)] for _ in range(4)]
    for br in range(2):
        for bc in range(2):
            color = base[br][bc]
            for dr in range(2):
                for dc in range(2):
                    tile[br * 2 + dr][bc * 2 + dc] = color
    return tile


def quad_channel_names() -> list[str]:
    return [f"ch{i}" for i in range(16)]


def tile_colors(tile: list[list[str]]) -> list[str]:
    return [tile[r][c] for r in range(len(tile)) for c in range(len(tile[0]))]


def _crop_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    h = x.shape[0] - x.shape[0] % multiple
    w = x.shape[1] - x.shape[1] % multiple
    return x[:h, :w, ...]


def _mosaic(rgb: torch.Tensor, tile: list[list[str]]) -> torch.Tensor:
    scale = len(tile)
    rgb = _crop_multiple(rgb, scale)
    out = torch.empty((*rgb.shape[:2], 1), dtype=rgb.dtype, device=rgb.device)
    for r in range(scale):
        for c in range(scale):
            out[r::scale, c::scale, 0] = rgb[r::scale, c::scale, COLOR_INDEX[tile[r][c]]]
    return out


def _pack_mosaic(mosaic: torch.Tensor, scale: int) -> torch.Tensor:
    m = _crop_multiple(mosaic, scale)
    return torch.stack([m[r::scale, c::scale, 0] for r in range(scale) for c in range(scale)], dim=-1)


def _filter_packed_channels(packed: torch.Tensor, kernel_size: int, operation: str) -> torch.Tensor:
    x = packed.permute(2, 0, 1).unsqueeze(0)
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    x = F.pad(x, (pad_left, pad_right, pad_left, pad_right), mode="replicate")
    y = F.avg_pool2d(x, kernel_size=kernel_size, stride=1)
    if operation == "sum":
        y = y * float(kernel_size * kernel_size)
    elif operation != "average":
        raise ValueError(f"Unsupported binning_operation: {operation}")
    return y.squeeze(0).permute(1, 2, 0).contiguous()


def make_raw(camera_rgb: torch.Tensor, camera: dict[str, Any]) -> tuple[torch.Tensor, list[str], list[Any], int]:
    cfa = camera["cfa"]
    readout = camera.get("readout", {"mode": "full"})
    cfa_type = cfa.get("type", "bayer").lower().replace("_", "")
    pattern = cfa.get("pattern", "RGGB").upper()

    if cfa_type == "quadbayer":
        tile = quad_tile(pattern)
        mosaic = _mosaic(camera_rgb, tile)
        packed = _pack_mosaic(mosaic, 4)
        order = [
            {"channel": f"ch{i}", "tile_row": i // 4, "tile_col": i % 4, "color": color}
            for i, color in enumerate(tile_colors(tile))
        ]
        return packed, quad_channel_names(), order, 1

    if cfa_type not in {"bayer", "binning"}:
        raise ValueError(f"Unsupported cfa.type '{cfa.get('type')}'. Use 'bayer', 'quadbayer', or 'binning'.")

    tile = bayer_tile(pattern)
    mosaic = _mosaic(camera_rgb, tile)
    packed = _pack_mosaic(mosaic, 2)
    samples_per_channel = 1
    if cfa_type == "binning":
        packed = _filter_packed_channels(packed, kernel_size=2, operation=readout.get("binning_operation", "average"))
        samples_per_channel = 4
    names = bayer_channel_names(pattern)
    return packed, names, names, samples_per_channel
