"""
debug_extract.py
Debug version to see what objects are being detected
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import remove
from skimage.measure import label, regionprops


def remove_background(src_path: Path) -> Image.Image:
    """Return a transparent (RGBA) PIL image with the background removed."""
    with src_path.open("rb") as f:
        bg_removed = remove(f.read())          # rembg does the heavy lifting
    img = Image.open(io.BytesIO(bg_removed))  # load from bytes
    return img.convert("RGBA")


def mask_from_alpha(img: Image.Image) -> np.ndarray:
    """Return a boolean mask where True means 'object pixel'."""
    alpha = np.asarray(img)[:, :, 3]
    return alpha > 0


def debug_extract_and_save_objects(
    img: Image.Image,
    mask: np.ndarray,
    out_dir: Path,
    min_area: int = 300,
) -> None:
    """Debug version: Detect connected components, crop, and save each object with info."""
    labeled = label(mask, connectivity=2)
    regions = [r for r in regionprops(labeled) if r.area >= min_area]

    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(regions)} regions with area >= {min_area}")
    print("Region details:")
    for idx, r in enumerate(regions, start=1):
        print(f"  Region {idx}: area={r.area}, bbox={r.bbox}, centroid={r.centroid}")
        
        # Count non-transparent pixels in the cropped region
        y0, x0, y1, x1 = r.bbox
        cropped = img.crop((x0, y0, x1, y1))
        cropped_array = np.asarray(cropped)
        non_transparent = np.sum(cropped_array[:, :, 3] > 0)
        print(f"    Non-transparent pixels: {non_transparent}")
        
        # Save with region info in filename
        filename = f"object_{idx:02}_area{r.area}_pixels{non_transparent}.png"
        cropped.save(out_dir / filename)
        print(f"    Saved as: {filename}")

    print(f"✅  Saved {len(regions)} objects to {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug: Extract non-overlapping objects")
    parser.add_argument("image", type=Path, help="Input image file")
    parser.add_argument(
        "--min_area",
        type=int,
        default=300,
        help="Minimum component area (pixels) to keep",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("debug_objects"),
        help="Directory to save cropped objects",
    )
    args = parser.parse_args()

    print("🔄  Removing background…")
    rgba_img = remove_background(args.image)

    print("🔄  Generating mask & finding objects…")
    mask = mask_from_alpha(rgba_img)

    print("💾  Cropping and saving objects…")
    debug_extract_and_save_objects(rgba_img, mask, args.out_dir, args.min_area)


if __name__ == "__main__":
    main() 