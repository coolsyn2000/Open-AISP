from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .cfa import BAYER_PATTERNS, quad_tile, tile_colors
from .optics import apply_lens_psf_batch
from .sensor import max_dn, sensor_level_dn


COLOR_INDEX = {"R": 0, "Gr": 1, "Gb": 1, "G": 1, "B": 2}


def normalize_srgb_batch(rgb: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(rgb):
        x = rgb.to(torch.float32)
        if x.max() > 1.5:
            x = x / 255.0
    else:
        divisor = 65535.0 if rgb.dtype == torch.uint16 else 255.0
        x = rgb.to(torch.float32) / divisor
    return x.clamp(0.0, 1.0)


def _bayer_tile(pattern: str) -> list[list[str]]:
    return [list(row) for row in BAYER_PATTERNS[pattern.upper()]]


def _pack_mosaic(mosaic: torch.Tensor, scale: int) -> torch.Tensor:
    _, _, h, w = mosaic.shape
    h = h - h % scale
    w = w - w % scale
    mosaic = mosaic[:, :, :h, :w]
    return torch.cat([mosaic[:, :, row::scale, col::scale] for row in range(scale) for col in range(scale)], dim=1)


def _mosaic(rgb_nchw: torch.Tensor, tile: list[list[str]]) -> torch.Tensor:
    scale = len(tile)
    _, _, h, w = rgb_nchw.shape
    h = h - h % scale
    w = w - w % scale
    rgb_nchw = rgb_nchw[:, :, :h, :w]
    mosaic = torch.empty((rgb_nchw.shape[0], 1, h, w), dtype=rgb_nchw.dtype, device=rgb_nchw.device)
    for row in range(scale):
        for col in range(scale):
            mosaic[:, 0, row::scale, col::scale] = rgb_nchw[:, COLOR_INDEX[tile[row][col]], row::scale, col::scale]
    return mosaic


def _filter_binning_channels(packed: torch.Tensor, operation: str) -> torch.Tensor:
    x = F.pad(packed, (1, 0, 1, 0), mode="replicate")
    y = F.avg_pool2d(x, kernel_size=2, stride=1)
    if operation == "sum":
        y = y * 4.0
    elif operation != "average":
        raise ValueError(f"Unsupported binning_operation: {operation}")
    return y.contiguous()


def make_raw_batch(camera_rgb: torch.Tensor, camera: dict[str, Any]) -> tuple[torch.Tensor, list[str], int]:
    cfa = camera["cfa"]
    readout = camera.get("readout", {"mode": "full"})
    cfa_type = cfa.get("type", "bayer").lower().replace("_", "")
    pattern = cfa.get("pattern", "RGGB").upper()
    if cfa_type == "quadbayer":
        tile = quad_tile(pattern)
        return _pack_mosaic(_mosaic(camera_rgb, tile), 4), [f"ch{i}" for i in range(16)], 1
    if cfa_type not in {"bayer", "binning"}:
        raise ValueError(f"Unsupported cfa.type '{cfa.get('type')}'. Use 'bayer', 'quadbayer', or 'binning'.")
    tile = _bayer_tile(pattern)
    packed = _pack_mosaic(_mosaic(camera_rgb, tile), 2)
    if cfa_type == "binning":
        packed = _filter_binning_channels(packed, readout.get("binning_operation", "average"))
        return packed, [tile[0][0], tile[0][1], tile[1][0], tile[1][1]], 4
    return packed, [tile[0][0], tile[0][1], tile[1][0], tile[1][1]], 1


def apply_sensor_levels_batch(camera_rgb: torch.Tensor, camera: dict[str, Any]) -> torch.Tensor:
    sensor = camera["sensor"]
    bit_depth = int(sensor["bit_depth"])
    black = sensor_level_dn(sensor, "black_level") / max_dn(bit_depth)
    white = sensor_level_dn(sensor, "white_level") / max_dn(bit_depth)
    if not 0.0 <= black < white <= 1.0:
        raise ValueError(f"Invalid sensor black/white levels: {black}, {white}")
    x = black + camera_rgb.clamp(0.0, 1.0) * (white - black)
    if sensor.get("quantize", True):
        levels = float(max_dn(bit_depth))
        x = torch.round(x * levels) / levels
    return x.clamp(0.0, 1.0)


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


def _noise_vectors(
    clean: torch.Tensor,
    analog_gain: torch.Tensor,
    channel_names: list[str],
    camera: dict[str, Any],
    samples_per_channel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    calibration = camera["noise"]["calibration"]
    base_gain = float(calibration.get("base_analog_gain", 1.0))
    ratio = analog_gain.to(clean.dtype).view(-1, 1, 1, 1) / max(base_gain, 1e-8)
    shot_exp = float(calibration.get("shot_noise_gain_exponent", 1.0))
    read_exp = float(calibration.get("read_noise_gain_exponent", 1.0))
    base = calibration["channels"]
    k_base = torch.tensor([float(base[_lookup_channel(base, name, camera)]["K"]) for name in channel_names], dtype=clean.dtype, device=clean.device)
    sigma_base = torch.tensor(
        [float(base[_lookup_channel(base, name, camera)]["sigma_read"]) for name in channel_names],
        dtype=clean.dtype,
        device=clean.device,
    )
    k = k_base.view(1, -1, 1, 1) * ratio**shot_exp
    sigma = sigma_base.view(1, -1, 1, 1) * ratio**read_exp
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


def noise_std_map_batch(
    clean: torch.Tensor,
    analog_gain: torch.Tensor,
    channel_names: list[str],
    camera: dict[str, Any],
    samples_per_channel: int,
    reduce: str,
) -> torch.Tensor:
    sensor = camera["sensor"]
    bit_depth = int(sensor["bit_depth"])
    black = sensor_level_dn(sensor, "black_level") / max_dn(bit_depth)
    white = sensor_level_dn(sensor, "white_level") / max_dn(bit_depth)
    active = ((clean - black) / max(white - black, 1e-8)).clamp(0.0, 1.0)
    k, sigma = _noise_vectors(clean, analog_gain, channel_names, camera, samples_per_channel)
    var = (k * active + sigma.square()).clamp_min(1e-12) * float(camera["noise"].get("strength_multiplier", 1.0))
    std = torch.sqrt(var)
    if reduce == "mean":
        return std.mean(dim=1, keepdim=True)
    if reduce == "max":
        return std.max(dim=1, keepdim=True).values
    if reduce == "none":
        return std
    raise ValueError(f"Unsupported noise_std_map reduce mode: {reduce}")


def srgb_to_raw_burst_batch(
    rgb: torch.Tensor,
    camera: dict[str, Any],
    analog_gain: torch.Tensor,
    seed: torch.Tensor | None = None,
    frames: int = 3,
    noise_map_reduce: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if noise_map_reduce == "none":
        raise ValueError("srgb_to_raw_burst_batch expects a single-channel noise map; use noise_map_reduce='mean' or 'max'")

    rgb = normalize_srgb_batch(rgb).to(analog_gain.device, non_blocking=True)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected BHWC RGB batch, got {tuple(rgb.shape)}")
    target_hwc = rgb.pow(float(camera["gamma"]))
    target = target_hwc.permute(0, 3, 1, 2).contiguous()

    ccm = torch.tensor(camera["ccm"]["matrix"], dtype=target_hwc.dtype, device=target_hwc.device)
    inv_ccm = torch.linalg.inv(ccm)
    camera_rgb_hwc = target_hwc @ inv_ccm.T
    awb = camera["awb"]
    gains = torch.tensor(
        [float(awb["red_gain"]), float(awb.get("green_gain", 1.0)), float(awb["blue_gain"])],
        dtype=camera_rgb_hwc.dtype,
        device=camera_rgb_hwc.device,
    )
    camera_rgb = (camera_rgb_hwc / gains.view(1, 1, 1, 3).clamp_min(1e-8)).clamp(0.0, 1.0).permute(0, 3, 1, 2)
    camera_rgb, _ = apply_lens_psf_batch(camera_rgb, camera)
    sensor_rgb = apply_sensor_levels_batch(camera_rgb, camera)
    clean_raw, channel_names, samples_per_channel = make_raw_batch(sensor_rgb, camera)
    std = noise_std_map_batch(clean_raw, analog_gain, channel_names, camera, samples_per_channel, noise_map_reduce)

    gen = torch.Generator(device=clean_raw.device)
    if seed is not None and seed.numel() > 0:
        gen.manual_seed(int(seed[0].item()))
    noise = torch.randn(
        (clean_raw.shape[0], int(frames), clean_raw.shape[1], clean_raw.shape[2], clean_raw.shape[3]),
        generator=gen,
        dtype=clean_raw.dtype,
        device=clean_raw.device,
    )
    raw_frames = (clean_raw[:, None] + noise * std[:, None]).clamp(0.0, 1.0)
    x = torch.cat([raw_frames[:, frame_idx] for frame_idx in range(int(frames))] + [std], dim=1).to(torch.float32)

    scale = 4 if camera["cfa"].get("type", "bayer").lower().replace("_", "") == "quadbayer" else 2
    target = target[:, :, : clean_raw.shape[-2] * scale, : clean_raw.shape[-1] * scale].to(torch.float32)
    return x, target


def simulate_burst_batch_on_device(
    rgb: torch.Tensor,
    analog_gain: torch.Tensor,
    seed: torch.Tensor,
    camera: dict[str, Any],
    frames: int,
    noise_map_reduce: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return srgb_to_raw_burst_batch(rgb, camera, analog_gain, seed, frames, noise_map_reduce)
