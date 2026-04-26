from __future__ import annotations

from typing import Any

import torch

BASE_BLACK_LEVEL_10BIT = 32


def max_dn(bit_depth: int) -> int:
    return (1 << int(bit_depth)) - 1


def default_black_level_dn(bit_depth: int, reference_10bit: int = BASE_BLACK_LEVEL_10BIT) -> int:
    bit_depth = int(bit_depth)
    if bit_depth >= 10:
        return int(reference_10bit * (1 << (bit_depth - 10)))
    return int(round(reference_10bit / (1 << (10 - bit_depth))))


def sensor_level_dn(sensor: dict[str, Any], key: str) -> int:
    bit_depth = int(sensor["bit_depth"])
    value = sensor.get(key, "auto")
    if value in (None, "auto"):
        if key == "black_level":
            return default_black_level_dn(bit_depth, int(sensor.get("black_level_10bit_reference", BASE_BLACK_LEVEL_10BIT)))
        if key == "white_level":
            return max_dn(bit_depth)
    return int(round(float(value)))


def apply_sensor_levels(camera_rgb: torch.Tensor, sensor: dict[str, Any]) -> torch.Tensor:
    bit_depth = int(sensor["bit_depth"])
    black = sensor_level_dn(sensor, "black_level") / max_dn(bit_depth)
    white = sensor_level_dn(sensor, "white_level") / max_dn(bit_depth)
    if not 0.0 <= black < white <= 1.0:
        raise ValueError(f"Invalid sensor black/white levels: {black}, {white}")
    x = camera_rgb.clamp(0.0, 1.0)
    x = black + x * (white - black)
    if sensor.get("quantize", True):
        levels = float(max_dn(bit_depth))
        x = torch.round(x * levels) / levels
    return x.clamp(0.0, 1.0)

