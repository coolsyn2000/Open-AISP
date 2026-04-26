from __future__ import annotations

import argparse
import copy
import functools
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import make_dataset_from_config
from .losses import CharbonnierLoss
from .metrics import MetricComputer
from .model import build_model, infer_raw_channels
from .rawsim_bridge import ensure_raw_sim_importable
from .utils import cfa_scale, load_json, save_json, set_random_seed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

ensure_raw_sim_importable()

from raw_sim.config import load_camera_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train joint denoise+demosaic NAFNet on simulated RAW bursts.")
    parser.add_argument("--config", required=True, help="Training config JSON.")
    parser.add_argument("--camera-module-json", help="Override camera module JSON from the training config.")
    parser.add_argument("--resume", help="Checkpoint path.")
    return parser


def _to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["input"].to(device, non_blocking=True), batch["target"].to(device, non_blocking=True)


def init_data_worker(worker_id: int, torch_threads: int = 1) -> None:
    torch.set_num_threads(max(1, int(torch_threads)))
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_loader(dataset: torch.utils.data.Dataset, config: dict[str, Any], device: torch.device, shuffle: bool, drop_last: bool) -> DataLoader:
    train_cfg = config["train"]
    num_workers = int(train_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg.get("batch_size", 4)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": drop_last,
        "collate_fn": collate_batch,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
        kwargs["worker_init_fn"] = functools.partial(init_data_worker, torch_threads=int(train_cfg.get("worker_torch_threads", 1)))
    return DataLoader(dataset, **kwargs)


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input": torch.stack([s["input"] for s in samples], dim=0),
        "target": torch.stack([s["target"] for s in samples], dim=0),
        "iso": torch.stack([s["iso"] for s in samples], dim=0),
        "analog_gain": torch.stack([s["analog_gain"] for s in samples], dim=0),
        "metadata": [s["metadata"] for s in samples],
    }


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, epoch: int, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def resolve_config_paths(config: dict[str, Any], config_path: str | Path, camera_module_json: str | None = None) -> dict[str, Any]:
    out = copy.deepcopy(config)
    base_dir = Path(config_path).resolve().parent

    def resolve(value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        return str(path if path.is_absolute() else (base_dir / path).resolve())

    def resolve_roots(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return resolve(value)

    if camera_module_json is not None:
        override_path = Path(camera_module_json)
        out["camera_module_json"] = str(override_path if override_path.is_absolute() else override_path.resolve())
    if "camera_module_json" not in out:
        raise ValueError("Training config must define camera_module_json")
    out["camera_module_json"] = resolve(out["camera_module_json"])

    data = out.setdefault("data", {})
    if "image_roots" in data:
        data["image_roots"] = resolve_roots(data["image_roots"])
    if "image_root" in data:
        data["image_root"] = resolve(data["image_root"])
    for split in ("train", "val"):
        if split in data and "image_roots" in data[split]:
            data[split]["image_roots"] = resolve_roots(data[split]["image_roots"])
        if split in data and "image_root" in data[split]:
            data[split]["image_root"] = resolve(data[split]["image_root"])
    return out


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    metrics: MetricComputer,
    device: torch.device,
    max_batches: int,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    lpips_count = 0
    count = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, y = _to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x)
        loss_total += float(criterion(pred, y).item())
        result = metrics(pred, y)
        psnr_total += result.psnr
        ssim_total += result.ssim
        if not math.isnan(result.lpips):
            lpips_total += result.lpips
            lpips_count += 1
        count += 1
    model.train()
    return {
        "loss": loss_total / max(count, 1),
        "psnr": psnr_total / max(count, 1),
        "ssim": ssim_total / max(count, 1),
        "lpips": lpips_total / lpips_count if lpips_count > 0 else float("nan"),
    }


def train(config: dict[str, Any], resume: str | None = None) -> None:
    set_random_seed(int(config.get("seed", 0)))
    requested_device = config.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda":
        gpu_id = int(config.get("gpu_id", 0))
        if not torch.cuda.is_available():
            raise RuntimeError("device is set to 'cuda', but torch.cuda.is_available() is False")
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"gpu_id={gpu_id} is invalid; available GPU count is {torch.cuda.device_count()}")
        requested_device = f"cuda:{gpu_id}"
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["train"].get("cudnn_benchmark", True))
        torch.set_float32_matmul_precision(str(config["train"].get("matmul_precision", "high")))

    camera = load_camera_json(config["camera_module_json"])
    model = build_model(config, infer_raw_channels(camera), cfa_scale(camera)).to(device)
    criterion = CharbonnierLoss(float(config.get("loss", {}).get("charbonnier_eps", 1e-3)))
    metrics = MetricComputer(device, use_lpips=bool(config.get("metrics", {}).get("lpips", True)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("optimizer", {}).get("lr", 2e-4)),
        weight_decay=float(config.get("optimizer", {}).get("weight_decay", 0.0)),
    )
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        step = int(ckpt.get("step", 0))

    train_dataset = make_dataset_from_config(config, split="train")
    train_loader = make_loader(train_dataset, config, device, shuffle=bool(config["train"].get("shuffle_images", False)), drop_last=True)

    val_loader = None
    val_cfg = config.get("data", {}).get("val", {})
    if val_cfg.get("image_root") or val_cfg.get("image_roots"):
        val_dataset = make_dataset_from_config(config, split="val")
        val_loader = make_loader(val_dataset, config, device, shuffle=False, drop_last=False)

    out_dir = Path(config["train"].get("output_dir", "runs/jdd"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "train_config.json", config)
    log_path = out_dir / "train.log"
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "none"
    append_log(
        log_path,
        f"start device={device} gpu_name={gpu_name} amp={use_amp} "
        f"num_workers={config['train'].get('num_workers', 0)} "
        f"persistent_workers={config['train'].get('persistent_workers', True)} "
        f"prefetch_factor={config['train'].get('prefetch_factor', 4)} "
        f"camera_module_json={config['camera_module_json']}",
    )

    iterations = int(config["train"].get("iterations", config["train"].get("max_iterations", 100000)))
    log_every = int(config["train"].get("log_every", 20))
    save_every = int(config["train"].get("save_every", 500))
    val_every = int(config["train"].get("val_every", 500))
    max_val_batches = int(config["train"].get("max_val_batches", 10))

    model.train()
    t0 = time.time()
    epoch = start_epoch
    pbar_ctx = tqdm(total=iterations, initial=step, desc="train", dynamic_ncols=True) if tqdm is not None else None
    try:
        while step < iterations:
            epoch += 1
            for batch in train_loader:
                if step >= iterations:
                    break
                x, y = _to_device(batch, device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pred = model(x)
                    loss = criterion(pred, y)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"].get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                step += 1

                if pbar_ctx is not None:
                    pbar_ctx.update(1)
                    pbar_ctx.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}")

                if step % log_every == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    message = f"train epoch={epoch} step={step} loss={loss.item():.6f} samples_per_sec={step * x.shape[0] / elapsed:.2f}"
                    append_log(log_path, message)
                    if pbar_ctx is None:
                        print(message)

                if step % save_every == 0:
                    ckpt_path = out_dir / f"step_{step:08d}.pth"
                    save_checkpoint(ckpt_path, model, optimizer, step, epoch, config)
                    append_log(log_path, f"checkpoint step={step} path={ckpt_path}")

                if val_loader is not None and step % val_every == 0:
                    val = validate(model, val_loader, criterion, metrics, device, max_val_batches, use_amp)
                    message = (
                        f"val epoch={epoch} step={step} loss={val['loss']:.6f} "
                        f"psnr={val['psnr']:.4f} ssim={val['ssim']:.6f} lpips={val['lpips']:.6f}"
                    )
                    append_log(log_path, message)
                    if pbar_ctx is not None:
                        pbar_ctx.write(message)
                    else:
                        print(message)

            save_checkpoint(out_dir / "latest.pth", model, optimizer, step, epoch, config)
    finally:
        if pbar_ctx is not None:
            pbar_ctx.close()


def main() -> None:
    args = build_parser().parse_args()
    config = resolve_config_paths(load_json(args.config), args.config, args.camera_module_json)
    train(config, args.resume)


if __name__ == "__main__":
    main()
