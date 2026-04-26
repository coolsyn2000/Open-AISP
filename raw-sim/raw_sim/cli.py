from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_camera_json
from .images import collect_images
from .outputs import save_training_pair
from .pipeline import simulate_image


def generate_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    camera_json: str | Path,
    patch_size: int | None = None,
    num_patches: int | None = None,
    random_seed: int = 0,
) -> None:
    input_path = Path(input_path)
    images = collect_images(input_path)
    if not images:
        raise SystemExit(f"No RGB images found in {input_path}")
    generate_from_images(images, output_dir, camera_json, patch_size, num_patches, random_seed)


def generate_from_images(
    images: list[Path],
    output_dir: str | Path,
    camera_json: str | Path,
    patch_size: int | None = None,
    num_patches: int | None = None,
    random_seed: int = 0,
) -> None:
    output_dir = Path(output_dir)
    camera_json = Path(camera_json)
    camera = load_camera_json(camera_json)
    total = int(num_patches) if num_patches else len(images)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input images: {len(images)}")
    print(f"Output dir: {output_dir}")
    print(f"Camera JSON: {camera_json}")
    print(f"Samples: {total}")

    for sample_index in range(total):
        image_path = images[sample_index % len(images)]
        seed = int(random_seed) + sample_index
        result = simulate_image(image_path, camera, patch_size, seed)

        sample_dir = output_dir / f"{sample_index:08d}_{image_path.stem}"
        save_training_pair(
            sample_dir,
            result["input_noisy_raw"],
            result["gt_linear_rgb"],
            result["metadata"],
            stem="sample",
        )
        print(f"[{sample_index + 1}/{total}] {image_path.name} -> {sample_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate noisy RAW input and linear RGB GT pairs from RGB images.")
    parser.add_argument("--input", required=True, help="RGB image file or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--camera-json", required=True, help="Camera parameter JSON.")
    parser.add_argument("--patch-size", type=int, default=None, help="Random crop size. Omit to use full images.")
    parser.add_argument("--num-patches", type=int, default=None, help="Number of samples to generate. Omit to use each image once.")
    parser.add_argument("--random-seed", type=int, default=0, help="Base random seed.")
    args = parser.parse_args(argv)

    generate_dataset(
        input_path=args.input,
        output_dir=args.output,
        camera_json=args.camera_json,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
