from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import collect_rgb_images, simulate_burst_sample
from .metrics import MetricComputer
from .model import build_model, infer_raw_channels
from .rawsim_bridge import ensure_raw_sim_importable
from .train import resolve_config_paths
from .utils import cfa_scale, load_json, save_json
from .visualize import save_rgb_tensor

ensure_raw_sim_importable()

from raw_sim.config import load_camera_json  # noqa: E402
from raw_sim.sensor import max_dn, sensor_level_dn  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run JDD inference and save visualized predictions plus metrics.")
    parser.add_argument("--checkpoint", required=True, help="Model .pth checkpoint.")
    parser.add_argument("--config", help="Training config JSON. If omitted, use the config saved in checkpoint.")
    parser.add_argument("--camera-module-json", help="Override camera module JSON.")
    parser.add_argument("--input", nargs="+", required=True, help="RGB image files or directories.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--mode", choices=["full", "patch"], default="full")
    parser.add_argument("--patch-size", type=int, default=128, help="RGB patch size for patch inference.")
    parser.add_argument("--overlap", type=int, default=32, help="RGB overlap for patch inference.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    return parser


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


@torch.no_grad()
def patch_inference(model: torch.nn.Module, x: torch.Tensor, scale: int, patch_size: int, overlap: int) -> torch.Tensor:
    raw_tile = max(int(patch_size) // scale, 1)
    raw_overlap = max(int(overlap) // scale, 0)
    stride = max(raw_tile - raw_overlap, 1)
    _, _, raw_h, raw_w = x.shape
    out_h, out_w = raw_h * scale, raw_w * scale
    output = torch.zeros((1, 3, out_h, out_w), dtype=x.dtype, device=x.device)
    weight = torch.zeros_like(output)

    for top in _positions(raw_h, raw_tile, stride):
        for left in _positions(raw_w, raw_tile, stride):
            crop = x[:, :, top : top + raw_tile, left : left + raw_tile]
            pred = model(crop)
            y0, y1 = top * scale, (top + crop.shape[-2]) * scale
            x0, x1 = left * scale, (left + crop.shape[-1]) * scale
            output[:, :, y0:y1, x0:x1] += pred
            weight[:, :, y0:y1, x0:x1] += 1.0
    return output / weight.clamp_min(1.0)


def _load_config(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if args.config:
        return resolve_config_paths(load_json(args.config), args.config, args.camera_module_json)
    config = dict(checkpoint["config"])
    if args.camera_module_json:
        config["camera_module_json"] = str(Path(args.camera_module_json).resolve())
    else:
        camera_path = Path(config["camera_module_json"])
        if not camera_path.exists():
            camera_name = PureWindowsPath(str(config["camera_module_json"])).name
            local_camera_path = Path(__file__).resolve().parents[1] / "configs" / camera_name
            if local_camera_path.exists():
                config["camera_module_json"] = str(local_camera_path)
    return config


def _comparison_image(pred: torch.Tensor, target: torch.Tensor) -> Image.Image:
    pred_arr = np.asarray(save_preview_image(pred))
    target_arr = np.asarray(save_preview_image(target))
    error = torch.abs(pred.detach().cpu().clamp(0, 1) - target.detach().cpu().clamp(0, 1))
    error = error[0].permute(1, 2, 0).numpy()
    error = np.round(np.clip(error * 4.0, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(np.concatenate([target_arr, pred_arr, error], axis=1), mode="RGB")


def save_preview_image(x: torch.Tensor) -> Image.Image:
    if x.ndim == 4:
        x = x[0]
    arr = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = np.round(arr ** (1.0 / 2.2) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _opencv_bayer_code(pattern: str) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV baseline requires opencv-python. Install it with `pip install opencv-python`.") from exc

    codes = {
        "RGGB": cv2.COLOR_BayerBG2RGB,
        "BGGR": cv2.COLOR_BayerRG2RGB,
        "GRBG": cv2.COLOR_BayerGB2RGB,
        "GBRG": cv2.COLOR_BayerGR2RGB,
    }
    pattern = pattern.upper()
    if pattern not in codes:
        raise ValueError(f"Unsupported Bayer pattern for OpenCV demosaic: {pattern}")
    return codes[pattern]


def _unpack_raw_frame(raw_chw: torch.Tensor, scale: int) -> torch.Tensor:
    channels, raw_h, raw_w = raw_chw.shape
    if channels != scale * scale:
        raise ValueError(f"Expected {scale * scale} RAW channels, got {channels}")
    mosaic = torch.empty((raw_h * scale, raw_w * scale), dtype=raw_chw.dtype, device=raw_chw.device)
    ch = 0
    for row in range(scale):
        for col in range(scale):
            mosaic[row::scale, col::scale] = raw_chw[ch]
            ch += 1
    return mosaic


def opencv_demosaic_baseline(x: torch.Tensor, camera: dict[str, Any], raw_channels: int, scale: int) -> torch.Tensor:
    import cv2

    base_frame = x[0, :raw_channels].detach().cpu()
    mosaic = _unpack_raw_frame(base_frame, scale)

    sensor = camera["sensor"]
    bit_depth = int(sensor["bit_depth"])
    black = sensor_level_dn(sensor, "black_level") / max_dn(bit_depth)
    white = sensor_level_dn(sensor, "white_level") / max_dn(bit_depth)
    active = ((mosaic - black) / max(white - black, 1e-8)).clamp(0.0, 1.0)
    active_u16 = np.round(active.numpy() * 65535.0).astype(np.uint16)

    rgb = cv2.cvtColor(active_u16, _opencv_bayer_code(camera["cfa"].get("pattern", "RGGB"))).astype(np.float32) / 65535.0
    cam_rgb = torch.from_numpy(rgb)

    awb = camera["awb"]
    gains = torch.tensor(
        [float(awb["red_gain"]), float(awb.get("green_gain", 1.0)), float(awb["blue_gain"])],
        dtype=cam_rgb.dtype,
    )
    ccm = torch.tensor(camera["ccm"]["matrix"], dtype=cam_rgb.dtype)
    linear = (cam_rgb * gains.view(1, 1, 3)) @ ccm.T
    return linear.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0)


def _comparison_with_baseline(pred: torch.Tensor, baseline: torch.Tensor, target: torch.Tensor) -> Image.Image:
    target_arr = np.asarray(save_preview_image(target))
    pred_arr = np.asarray(save_preview_image(pred))
    baseline_arr = np.asarray(save_preview_image(baseline))
    pred_error = torch.abs(pred.detach().cpu().clamp(0, 1) - target.detach().cpu().clamp(0, 1))[0].permute(1, 2, 0).numpy()
    base_error = torch.abs(baseline.detach().cpu().clamp(0, 1) - target.detach().cpu().clamp(0, 1))[0].permute(1, 2, 0).numpy()
    pred_error = np.round(np.clip(pred_error * 4.0, 0, 1) * 255.0).astype(np.uint8)
    base_error = np.round(np.clip(base_error * 4.0, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(np.concatenate([target_arr, pred_arr, baseline_arr, pred_error, base_error], axis=1), mode="RGB")


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = _load_config(args, ckpt)
    camera = load_camera_json(config["camera_module_json"])
    scale = cfa_scale(camera)
    raw_channels = infer_raw_channels(camera)
    model = build_model(config, raw_channels, scale).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    images = collect_rgb_images(args.input)
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise SystemExit("No RGB images found for inference")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "inference_config.json", {"config": config, "checkpoint": str(Path(args.checkpoint).resolve()), "mode": args.mode})
    metrics = MetricComputer(device, use_lpips=bool(config.get("metrics", {}).get("lpips", True)))
    rows = []

    for idx, image_path in enumerate(images):
        sample = simulate_burst_sample(
            image_path=image_path,
            camera_module=camera,
            patch_size=None,
            seed=int(args.seed) + idx,
            frames=int(config.get("data", {}).get("frames", 3)),
            noise_map_reduce=config.get("data", {}).get("noise_map_reduce", "mean"),
        )
        x = sample["input"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)
        if args.mode == "patch":
            pred = patch_inference(model, x, scale, args.patch_size, args.overlap)
        else:
            pred = model(x)
        pred = pred[:, :, : target.shape[-2], : target.shape[-1]]
        opencv_pred = opencv_demosaic_baseline(x.cpu(), camera, raw_channels, scale).to(device)
        opencv_pred = opencv_pred[:, :, : target.shape[-2], : target.shape[-1]]
        result = metrics(pred, target)
        opencv_result = metrics(opencv_pred, target)

        stem = f"{idx:04d}_{Path(image_path).stem}"
        sample_dir = out_dir / f"{idx:04d}_{Path(image_path).stem}"
        save_rgb_tensor(sample_dir / "pred.png", pred)
        save_rgb_tensor(sample_dir / "opencv_demosaic.png", opencv_pred)
        save_rgb_tensor(sample_dir / "gt.png", target)
        _comparison_image(pred, target).save(sample_dir / "comparison_gt_pred_error.png")
        _comparison_with_baseline(pred, opencv_pred, target).save(sample_dir / "comparison_gt_jdd_opencv_errors.png")

        save_json(sample_dir / "metadata.json", sample["metadata"])
        row = {
            "image": str(Path(image_path).resolve()),
            "psnr": result.psnr,
            "ssim": result.ssim,
            "lpips": result.lpips,
            "opencv_psnr": opencv_result.psnr,
            "opencv_ssim": opencv_result.ssim,
            "opencv_lpips": opencv_result.lpips,
            "analog_gain": float(sample["analog_gain"].item()),
            "iso": int(sample["iso"].item()),
        }
        rows.append(row)
        save_json(sample_dir / "metrics.json", row)
        print(
            f"{stem}: jdd_psnr={result.psnr:.4f} jdd_ssim={result.ssim:.6f} jdd_lpips={result.lpips:.6f} "
            f"opencv_psnr={row['opencv_psnr']:.4f} opencv_ssim={row['opencv_ssim']:.6f} opencv_lpips={row['opencv_lpips']:.6f}"
        )

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "psnr", "ssim", "lpips", "opencv_psnr", "opencv_ssim", "opencv_lpips", "analog_gain", "iso"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    run_inference(build_parser().parse_args())


if __name__ == "__main__":
    main()
