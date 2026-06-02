"""
made by AI
"""

import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from datasets import load_dataset


IMAGE_SIZE = 32
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
HF_CACHE_DIR = DATA_DIR / "ffhq" / "hf_cache"
OUTPUT_DIR = DATA_DIR / "ffhq" / f"{IMAGE_SIZE}x{IMAGE_SIZE}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-token",
        type=str,
        # default="hf token",
        help="Hugging Face token. If omitted, HF_TOKEN is used if set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "Dmini/FFHQ-64x64",
        split="train",
        cache_dir=str(HF_CACHE_DIR),
        token=hf_token,
    )

    for idx in tqdm(range(len(dataset)), desc=f"Saving FFHQ {IMAGE_SIZE}x{IMAGE_SIZE}"):
        output_path = OUTPUT_DIR / f"{idx:06d}.png"
        if output_path.exists():
            continue

        tmp_path = output_path.with_suffix(".tmp")
        image = dataset[idx]["image"].convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        image.save(tmp_path, format="PNG")
        tmp_path.replace(output_path)

    print(f"Saved {len(dataset)} images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
