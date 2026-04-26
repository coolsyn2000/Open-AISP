from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_analog_gain(
    generator: torch.Generator,
    min_gain: float,
    max_gain: float,
    mode: str = "log_uniform",
) -> float:
    min_gain = max(float(min_gain), 1e-8)
    max_gain = max(float(max_gain), min_gain)
    u = torch.rand((), generator=generator).item()
    if mode == "uniform":
        return min_gain + (max_gain - min_gain) * u
    if mode == "log_uniform":
        return math.exp(math.log(min_gain) + (math.log(max_gain) - math.log(min_gain)) * u)
    raise ValueError(f"Unsupported analog_gain_sampling mode: {mode}")


def analog_gain_range(camera: dict[str, Any]) -> tuple[float, float, str]:
    camera_meta = camera.get("camera", {})
    value = camera_meta.get("analog_gain", 1.0)
    sampling = str(camera_meta.get("analog_gain_sampling", "log_uniform"))
    if isinstance(value, dict):
        return float(value.get("min", 1.0)), float(value.get("max", 1.0)), sampling
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("camera.analog_gain list must be [min, max]")
        return float(value[0]), float(value[1]), sampling
    gain = float(value)
    return gain, gain, sampling


def camera_with_analog_gain(camera: dict[str, Any], analog_gain: float) -> dict[str, Any]:
    out = deepcopy(camera)
    out.setdefault("camera", {})
    out["camera"]["analog_gain"] = float(analog_gain)
    out["camera"]["iso_definition"] = "ISO = analog_gain * 100"
    return out


def cfa_scale(camera: dict[str, Any]) -> int:
    cfa_type = camera["cfa"].get("type", "bayer").lower().replace("_", "")
    return 4 if cfa_type == "quadbayer" else 2
