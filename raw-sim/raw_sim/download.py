from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from .cli import generate_from_images
from .images import collect_images

DIV2K_URLS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}
FLICKR2K_FIRST_ROWS_URL = "https://datasets-server.huggingface.co/first-rows?dataset=yangtao9009%2FFlickr2K&config=default&split=train"
FLICKR2K_ZIP_URL = "https://huggingface.co/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip"
FLICKR2K_EXPECTED_IMAGES = 2650
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _short_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.netloc == "datasets-server.huggingface.co" and "/assets/" in parsed.path:
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return url


def _download(url: str, path: Path, skip_existing: bool = True, retries: int = 3, timeout: int = 60) -> None:
    if skip_existing and path.exists() and path.stat().st_size > 0:
        print(f"Using existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            print(f"Downloading {_short_url(url)} (attempt {attempt}/{retries})")
            request = urllib.request.Request(url, headers={"User-Agent": "raw-sim/0.2", "Connection": "close"})
            with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as f:
                shutil.copyfileobj(response, f)
            tmp.replace(path)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def _safe_path(root: Path, name: str) -> Path:
    resolved_root = root.resolve()
    target = (root / name).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        raise RuntimeError(f"Unsafe archive path: {name}")
    return target


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _remove_first_rows_preview(image_root: Path) -> bool:
    images = sorted(p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    preview_names = {f"{idx:06d}.jpg" for idx in range(100)}
    if len(images) != 100 or {p.name for p in images} != preview_names:
        return False
    for image in images:
        image.unlink()
    return True


def download_div2k(root: Path, splits: list[str], skip_existing: bool = True) -> Path:
    dataset_root = root / "DIV2K"
    archive_root = root / "archives"
    for split in splits:
        archive = archive_root / f"DIV2K_{split}_HR.zip"
        _download(DIV2K_URLS[split], archive, skip_existing)
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                target = _safe_path(dataset_root, info.filename)
                if skip_existing and target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return dataset_root


def _is_first_rows_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    return parsed.netloc == "datasets-server.huggingface.co" and parsed.path.rstrip("/") == "/first-rows"


def _download_flickr2k_first_rows(root: Path, image_root: Path, skip_existing: bool, url: str) -> None:
    manifest = root / "archives" / "Flickr2K_first_rows.json"
    _download(url, manifest, skip_existing=False)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not rows:
        raise RuntimeError("Flickr2K first-rows response has no rows")
    for row in rows:
        idx = int(row["row_idx"])
        src = row.get("row", {}).get("image", {}).get("src")
        if not src:
            continue
        suffix = Path(urllib.parse.urlsplit(src).path).suffix.lower() or ".jpg"
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".jpg"
        _download(src, image_root / f"{idx:06d}{suffix}", skip_existing)


def _extract_flickr2k_zip(archive: Path, image_root: Path, skip_existing: bool) -> int:
    extracted = 0
    with zipfile.ZipFile(archive) as zf:
        image_infos = [
            info for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_infos:
            raise RuntimeError(f"No image files found in Flickr2K archive: {archive}")
        for info in image_infos:
            target = _safe_path(image_root, Path(info.filename).name)
            if skip_existing and target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    return extracted


def download_flickr2k(root: Path, skip_existing: bool = True, url: str = FLICKR2K_ZIP_URL) -> Path:
    dataset_root = root / "Flickr2K"
    image_root = dataset_root / "Flickr2K_HR"
    image_root.mkdir(parents=True, exist_ok=True)

    if _is_first_rows_url(url):
        if skip_existing and _count_images(image_root) >= 100:
            print(f"Using existing Flickr2K preview images: {image_root}")
            return dataset_root
        _download_flickr2k_first_rows(root, image_root, skip_existing, url)
        return dataset_root

    if skip_existing and _count_images(image_root) >= FLICKR2K_EXPECTED_IMAGES:
        print(f"Using existing Flickr2K images: {image_root}")
        return dataset_root

    if skip_existing and _remove_first_rows_preview(image_root):
        print(f"Removed existing Flickr2K preview images before full download: {image_root}")

    archive_name = Path(urllib.parse.urlsplit(url).path).name or "Flickr2K.zip"
    archive = root / "archives" / archive_name
    _download(url, archive, skip_existing)
    extracted = _extract_flickr2k_zip(archive, image_root, skip_existing)
    image_count = _count_images(image_root)
    print(f"Flickr2K images ready: {image_count} files in {image_root} ({extracted} extracted)")
    if image_count < FLICKR2K_EXPECTED_IMAGES:
        print(f"Warning: expected about {FLICKR2K_EXPECTED_IMAGES} Flickr2K images, found {image_count}")
    return dataset_root


def simulate_from_roots(roots: list[Path], output: Path, camera_json: str, patch_size: int | None, num_patches: int | None, seed: int) -> None:
    images = []
    for root in roots:
        images.extend(collect_images(root))
    if not images:
        raise SystemExit("No downloaded images found for simulation")
    generate_from_images(images, output, camera_json, patch_size, num_patches, seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download HR RGB datasets and optionally generate noisy RAW / linear RGB pairs.")
    parser.add_argument("--dataset", choices=["div2k", "flickr2k", "flicker2k", "all"], default="all")
    parser.add_argument("--root", default="./datasets")
    parser.add_argument("--div2k-splits", nargs="+", choices=["train", "valid"], default=["train"])
    parser.add_argument("--flickr2k-url", default=FLICKR2K_ZIP_URL)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simulate-output")
    parser.add_argument("--camera-json")
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--num-patches", type=int)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    dataset = "flickr2k" if args.dataset == "flicker2k" else args.dataset
    roots = []
    if dataset in {"div2k", "all"}:
        roots.append(download_div2k(root, args.div2k_splits, args.skip_existing))
    if dataset in {"flickr2k", "all"}:
        roots.append(download_flickr2k(root, args.skip_existing, args.flickr2k_url))
    if args.simulate_output:
        if not args.camera_json:
            raise SystemExit("--camera-json is required with --simulate-output")
        simulate_from_roots(roots, Path(args.simulate_output), args.camera_json, args.patch_size, args.num_patches, args.random_seed)


if __name__ == "__main__":
    main()
