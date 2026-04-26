from __future__ import annotations

from typing import Any

import torch

from .cfa import quad_tile, tile_colors
from .sensor import max_dn, sensor_level_dn


def analog_gain(camera: dict[str, Any]) -> float:
    camera_meta = camera.get("camera", {})
    return float(camera_meta.get("analog_gain", camera.get("sensor", {}).get("analog_gain", 1.0)))


def iso_value(camera: dict[str, Any]) -> int:
    return int(round(analog_gain(camera) * 100.0))


def _lookup_channel(base: dict[str, dict[str, float]], name: str, camera: dict[str, Any]) -> str:
    if name in base:
        return name
    if name.startswith("ch"):
        idx = int(name[2:])
        color = tile_colors(quad_tile(camera["cfa"].get("pattern", "RGGB")))[idx]
        if color in base:
            return color
        if color in {"Gr", "Gb"} and "G" in base:
            return "G"
    if name in {"Gr", "Gb"} and "G" in base:
        return "G"
    raise KeyError(f"noise.calibration.channels missing '{name}'")


def noise_params(camera: dict[str, Any], channel_names: list[str]) -> dict[str, dict[str, float]]:
    calibration = camera["noise"]["calibration"]
    base_gain = float(calibration.get("base_analog_gain", 1.0))
    ratio = analog_gain(camera) / max(base_gain, 1e-8)
    shot_exp = float(calibration.get("shot_noise_gain_exponent", 1.0))
    read_exp = float(calibration.get("read_noise_gain_exponent", 1.0))
    base = calibration["channels"]
    params = {}
    for name in channel_names:
        p = base[_lookup_channel(base, name, camera)]
        params[name] = {
            "K": float(p["K"]) * ratio**shot_exp,
            "sigma_read": float(p["sigma_read"]) * ratio**read_exp,
        }
    return params


def _noise_vectors(
    clean: torch.Tensor,
    params: dict[str, dict[str, float]],
    channel_names: list[str],
    camera: dict[str, Any],
    samples_per_channel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.tensor([params[name]["K"] for name in channel_names], dtype=clean.dtype, device=clean.device).view(1, 1, -1)
    sigma = torch.tensor([params[name]["sigma_read"] for name in channel_names], dtype=clean.dtype, device=clean.device).view(1, 1, -1)
    readout = camera.get("readout", {})
    if camera["cfa"].get("type", "").lower().replace("_", "") == "binning":
        operation = readout.get("binning_operation", "average")
        divisor = samples_per_channel if operation == "average" else 1
        if operation == "average":
            k = k / float(samples_per_channel)
        if readout.get("binning_type") == "digital_binning":
            sigma = sigma * (samples_per_channel**0.5) / float(divisor)
        elif readout.get("binning_type") == "analog_binning":
            sigma = sigma / float(divisor)
    return k, sigma


def active_signal_from_raw(raw: torch.Tensor, camera: dict[str, Any]) -> torch.Tensor:
    sensor = camera["sensor"]
    bit_depth = int(sensor["bit_depth"])
    black = sensor_level_dn(sensor, "black_level") / max_dn(bit_depth)
    white = sensor_level_dn(sensor, "white_level") / max_dn(bit_depth)
    return ((raw - black) / max(white - black, 1e-8)).clamp(0.0, 1.0)


def noise_std_map(
    clean: torch.Tensor,
    params: dict[str, dict[str, float]],
    channel_names: list[str],
    camera: dict[str, Any],
    samples_per_channel: int,
    reduce: str = "mean",
) -> torch.Tensor:
    k, sigma = _noise_vectors(clean, params, channel_names, camera, samples_per_channel)
    strength = float(camera["noise"].get("strength_multiplier", 1.0))
    active = active_signal_from_raw(clean, camera)
    var = (k * active + sigma.square()).clamp_min(1e-12) * strength
    std = torch.sqrt(var)
    if reduce == "none":
        return std
    if reduce == "mean":
        return std.mean(dim=-1, keepdim=True)
    if reduce == "max":
        return std.max(dim=-1, keepdim=True).values
    raise ValueError(f"Unsupported noise_std_map reduce mode: {reduce}")


def simulate_noise(
    clean: torch.Tensor,
    params: dict[str, dict[str, float]],
    channel_names: list[str],
    camera: dict[str, Any],
    samples_per_channel: int,
    seed: int | None,
) -> torch.Tensor:
    gen = torch.Generator(device=clean.device if clean.is_cuda else "cpu")
    if seed is not None:
        gen.manual_seed(int(seed))
    std = noise_std_map(clean, params, channel_names, camera, samples_per_channel, reduce="none")
    noise = torch.randn(clean.shape, generator=gen, device=clean.device, dtype=clean.dtype) * std
    return (clean + noise).clamp(0.0, 1.0)
