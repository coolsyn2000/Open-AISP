from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .rawsim_bridge import ensure_raw_sim_importable
from .utils import analog_gain_range, camera_with_analog_gain, cfa_scale, sample_analog_gain

ensure_raw_sim_importable()

from raw_sim.cfa import make_raw  # noqa: E402
from raw_sim.config import load_camera_json  # noqa: E402
from raw_sim.images import collect_images  # noqa: E402
from raw_sim.noise import iso_value, noise_params, noise_std_map, simulate_noise  # noqa: E402
from raw_sim.optics import apply_lens_psf  # noqa: E402
from raw_sim.sensor import apply_sensor_levels  # noqa: E402
from raw_sim.unprocess import linear_rgb_gt_from_srgb, srgb_to_camera_rgb  # noqa: E402


@dataclass(frozen=True)
class BurstSample:
    input: torch.Tensor
    target: torch.Tensor
    iso: torch.Tensor
    analog_gain: torch.Tensor
    metadata: dict[str, Any]


def collect_rgb_images(image_roots: str | Path | Sequence[str | Path]) -> list[Path]:
    roots: Sequence[str | Path]
    if isinstance(image_roots, (str, Path)):
        roots = [image_roots]
    else:
        roots = image_roots
    images: list[Path] = []
    for root in roots:
        images.extend(collect_images(root))
    return sorted(dict.fromkeys(images))


def load_rgb_patch(path: str | Path, patch_size: int | None, seed: int) -> tuple[torch.Tensor, dict[str, int]]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if patch_size is None or patch_size <= 0:
            arr = np.asarray(img)
            return torch.from_numpy(arr.copy()), {"top": 0, "left": 0, "height": h, "width": w}

        patch_size = int(patch_size)
        if h < patch_size or w < patch_size:
            scale = patch_size / min(h, w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            w, h = img.size

        gen = torch.Generator().manual_seed(int(seed))
        top = int(torch.randint(0, h - patch_size + 1, (), generator=gen).item()) if h > patch_size else 0
        left = int(torch.randint(0, w - patch_size + 1, (), generator=gen).item()) if w > patch_size else 0
        patch = img.crop((left, top, left + patch_size, top + patch_size))
        arr = np.asarray(patch)
        return torch.from_numpy(arr.copy()), {"top": top, "left": left, "height": patch_size, "width": patch_size}


def crop_rgb_array(arr: np.ndarray, patch_size: int | None, seed: int) -> tuple[torch.Tensor, dict[str, int]]:
    h, w = arr.shape[:2]
    if patch_size is None or patch_size <= 0:
        return torch.from_numpy(arr.copy()), {"top": 0, "left": 0, "height": h, "width": w}

    patch_size = int(patch_size)
    if h < patch_size or w < patch_size:
        image = Image.fromarray(arr, mode="RGB")
        scale = patch_size / min(h, w)
        image = image.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BICUBIC)
        arr = np.asarray(image)
        h, w = arr.shape[:2]

    gen = torch.Generator().manual_seed(int(seed))
    top = int(torch.randint(0, h - patch_size + 1, (), generator=gen).item()) if h > patch_size else 0
    left = int(torch.randint(0, w - patch_size + 1, (), generator=gen).item()) if w > patch_size else 0
    patch = arr[top : top + patch_size, left : left + patch_size, :]
    return torch.from_numpy(patch.copy()), {"top": top, "left": left, "height": patch_size, "width": patch_size}


def simulate_burst_sample(
    image_path: str | Path,
    camera_module: dict[str, Any],
    patch_size: int | None,
    seed: int,
    frames: int = 3,
    noise_map_reduce: str = "mean",
    analog_gain: float | None = None,
    rgb_patch: torch.Tensor | None = None,
    crop: dict[str, int] | None = None,
) -> dict[str, Any]:
    if frames != 3:
        raise ValueError("This JDD framework expects exactly 3 noisy RAW frames")

    gen = torch.Generator().manual_seed(int(seed))
    if analog_gain is None:
        min_gain, max_gain, gain_sampling = analog_gain_range(camera_module)
        analog_gain = sample_analog_gain(gen, min_gain, max_gain, gain_sampling)

    image_path = Path(image_path)
    if rgb_patch is None:
        patch, crop = load_rgb_patch(image_path, patch_size, seed)
    else:
        patch = rgb_patch
        crop = crop or {"top": 0, "left": 0, "height": int(patch.shape[0]), "width": int(patch.shape[1])}
    camera = camera_with_analog_gain(camera_module, float(analog_gain))

    target = linear_rgb_gt_from_srgb(patch, camera)
    camera_rgb, _ = srgb_to_camera_rgb(patch, camera)
    camera_rgb, psf_meta = apply_lens_psf(camera_rgb, camera)
    sensor_rgb = apply_sensor_levels(camera_rgb, camera["sensor"])
    clean_raw, channel_names, packing_order, samples_per_channel = make_raw(sensor_rgb, camera)
    params = noise_params(camera, channel_names)

    raw_frames = [
        simulate_noise(clean_raw, params, channel_names, camera, samples_per_channel, int(seed) * 100 + frame_idx)
        for frame_idx in range(frames)
    ]
    noise_map = noise_std_map(clean_raw, params, channel_names, camera, samples_per_channel, reduce=noise_map_reduce)

    packed_h, packed_w = int(clean_raw.shape[0]), int(clean_raw.shape[1])
    scale = cfa_scale(camera)
    target = target[: packed_h * scale, : packed_w * scale, :]

    x = torch.cat([frame.permute(2, 0, 1) for frame in raw_frames] + [noise_map.permute(2, 0, 1)], dim=0).to(torch.float32)
    y = target.permute(2, 0, 1).contiguous().to(torch.float32)
    iso = torch.tensor(float(iso_value(camera)), dtype=torch.float32)
    gain = torch.tensor(float(analog_gain), dtype=torch.float32)
    metadata = {
        "source_image_path": str(image_path.resolve()),
        "crop_position": crop,
        "analog_gain": float(analog_gain),
        "iso": int(iso.item()),
        "cfa_type": camera["cfa"].get("type", "bayer"),
        "cfa_pattern": camera["cfa"].get("pattern", "RGGB"),
        "channel_names": channel_names,
        "packing_order": packing_order,
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "lens_psf": psf_meta,
        "noise_params": params,
    }
    return {"input": x, "target": y, "iso": iso, "analog_gain": gain, "metadata": metadata}


class SimulatedRawBurstDataset(Dataset):
    def __init__(
        self,
        image_roots: str | Path | Sequence[str | Path],
        camera_module_json: str | Path,
        patch_size: int = 128,
        max_images: int | None = None,
        length: int | None = None,
        frames: int = 3,
        seed: int = 0,
        noise_map_reduce: str = "mean",
        deterministic: bool = False,
        cache_images: bool = False,
        max_cached_images: int | None = None,
    ) -> None:
        self.image_paths = collect_rgb_images(image_roots)
        if not self.image_paths:
            raise ValueError(f"No RGB images found in {image_roots}")
        limit = max_images if max_images is not None else length
        if limit is not None:
            self.image_paths = self.image_paths[: max(1, min(int(limit), len(self.image_paths)))]
        self.camera = load_camera_json(camera_module_json)
        self.patch_size = int(patch_size)
        self.frames = int(frames)
        if self.frames != 3:
            raise ValueError("This JDD framework expects exactly 3 noisy RAW frames")
        self.min_gain, self.max_gain, self.gain_sampling = analog_gain_range(self.camera)
        self.seed = int(seed)
        self.noise_map_reduce = noise_map_reduce
        self.deterministic = bool(deterministic)
        self.cache_images = bool(cache_images)
        self.max_cached_images = max_cached_images
        self._image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

        scale = cfa_scale(self.camera)
        if self.patch_size % scale != 0:
            raise ValueError(f"patch_size must be divisible by CFA scale {scale}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _cached_rgb_patch(self, image_path: Path, seed: int) -> tuple[torch.Tensor, dict[str, int]]:
        if not self.cache_images:
            return load_rgb_patch(image_path, self.patch_size, seed)
        arr = self._image_cache.get(image_path)
        if arr is None:
            with Image.open(image_path) as img:
                arr = np.asarray(img.convert("RGB")).copy()
            self._image_cache[image_path] = arr
            if self.max_cached_images is not None:
                while len(self._image_cache) > int(self.max_cached_images):
                    self._image_cache.popitem(last=False)
        else:
            self._image_cache.move_to_end(image_path)
        return crop_rgb_array(arr, self.patch_size, seed)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[int(index)]
        seed = self.seed + int(index) if self.deterministic else int(torch.randint(0, 2_147_483_647, ()).item())
        patch, crop = self._cached_rgb_patch(image_path, seed)
        return simulate_burst_sample(
            image_path=image_path,
            camera_module=self.camera,
            patch_size=self.patch_size,
            seed=seed,
            frames=self.frames,
            noise_map_reduce=self.noise_map_reduce,
            rgb_patch=patch,
            crop=crop,
        )


class RgbPatchDataset(Dataset):
    def __init__(
        self,
        image_roots: str | Path | Sequence[str | Path],
        camera_module_json: str | Path,
        patch_size: int = 128,
        max_images: int | None = None,
        length: int | None = None,
        frames: int = 3,
        seed: int = 0,
        deterministic: bool = False,
        cache_images: bool = False,
        max_cached_images: int | None = None,
    ) -> None:
        self.image_paths = collect_rgb_images(image_roots)
        if not self.image_paths:
            raise ValueError(f"No RGB images found in {image_roots}")
        limit = max_images if max_images is not None else length
        if limit is not None:
            self.image_paths = self.image_paths[: max(1, min(int(limit), len(self.image_paths)))]
        self.camera = load_camera_json(camera_module_json)
        self.patch_size = int(patch_size)
        self.frames = int(frames)
        if self.frames != 3:
            raise ValueError("This JDD framework expects exactly 3 noisy RAW frames")
        self.min_gain, self.max_gain, self.gain_sampling = analog_gain_range(self.camera)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.cache_images = bool(cache_images)
        self.max_cached_images = max_cached_images
        self._image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

        scale = cfa_scale(self.camera)
        if self.patch_size % scale != 0:
            raise ValueError(f"patch_size must be divisible by CFA scale {scale}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _cached_rgb_patch(self, image_path: Path, seed: int) -> tuple[torch.Tensor, dict[str, int]]:
        if not self.cache_images:
            return load_rgb_patch(image_path, self.patch_size, seed)
        arr = self._image_cache.get(image_path)
        if arr is None:
            with Image.open(image_path) as img:
                arr = np.asarray(img.convert("RGB")).copy()
            self._image_cache[image_path] = arr
            if self.max_cached_images is not None:
                while len(self._image_cache) > int(self.max_cached_images):
                    self._image_cache.popitem(last=False)
        else:
            self._image_cache.move_to_end(image_path)
        return crop_rgb_array(arr, self.patch_size, seed)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[int(index)]
        seed = self.seed + int(index) if self.deterministic else int(torch.randint(0, 2_147_483_647, ()).item())
        patch, crop = self._cached_rgb_patch(image_path, seed)
        gen = torch.Generator().manual_seed(seed)
        analog_gain = sample_analog_gain(gen, self.min_gain, self.max_gain, self.gain_sampling)
        return {
            "rgb": patch,
            "seed": torch.tensor(seed, dtype=torch.int64),
            "analog_gain": torch.tensor(float(analog_gain), dtype=torch.float32),
            "iso": torch.tensor(float(round(float(analog_gain) * 100.0)), dtype=torch.float32),
            "metadata": {
                "source_image_path": str(image_path.resolve()),
                "crop_position": crop,
                "analog_gain": float(analog_gain),
                "iso": int(round(float(analog_gain) * 100.0)),
            },
        }


def make_dataset_from_config(config: dict[str, Any], split: str = "train") -> SimulatedRawBurstDataset:
    data_cfg = config["data"]
    split_cfg = data_cfg.get(split, {})
    image_roots = split_cfg.get("image_roots", split_cfg.get("image_root", data_cfg.get("image_roots", data_cfg.get("image_root"))))
    max_images = split_cfg.get("max_images", split_cfg.get("length", data_cfg.get("max_images")))
    return SimulatedRawBurstDataset(
        image_roots=image_roots,
        camera_module_json=config["camera_module_json"],
        patch_size=int(data_cfg.get("patch_size", 128)),
        max_images=max_images,
        frames=int(data_cfg.get("frames", 3)),
        seed=int(data_cfg.get("seed", 0)) + (0 if split == "train" else 100000),
        noise_map_reduce=data_cfg.get("noise_map_reduce", "mean"),
        deterministic=split != "train",
        cache_images=bool(data_cfg.get("cache_images", False)),
        max_cached_images=data_cfg.get("max_cached_images"),
    )


def make_rgb_patch_dataset_from_config(config: dict[str, Any], split: str = "train") -> RgbPatchDataset:
    data_cfg = config["data"]
    split_cfg = data_cfg.get(split, {})
    image_roots = split_cfg.get("image_roots", split_cfg.get("image_root", data_cfg.get("image_roots", data_cfg.get("image_root"))))
    max_images = split_cfg.get("max_images", split_cfg.get("length", data_cfg.get("max_images")))
    return RgbPatchDataset(
        image_roots=image_roots,
        camera_module_json=config["camera_module_json"],
        patch_size=int(data_cfg.get("patch_size", 128)),
        max_images=max_images,
        frames=int(data_cfg.get("frames", 3)),
        seed=int(data_cfg.get("seed", 0)) + (0 if split == "train" else 100000),
        deterministic=split != "train",
        cache_images=bool(data_cfg.get("cache_images", False)),
        max_cached_images=data_cfg.get("max_cached_images"),
    )
