"""
extract_objects.py
Extract objects from images using various detection methods:
1. Background removal + connected components (original method)
2. Color-based segmentation
3. Edge-based detection
4. Threshold-based detection

Run:
    python extract_objects.py input_image.jpg --min_area 500
    python extract_objects.py input_image.jpg --method color --min_area 500
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
from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import watershed
from skimage.feature.peak import peak_local_max
from skimage.color import rgb2gray, rgb2hsv
from scipy import ndimage


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


def mask_from_color_threshold(img: Image.Image, threshold: float = 0.1) -> np.ndarray:
    """Create mask using color-based thresholding."""
    # Convert to HSV for better color separation
    img_array = np.asarray(img)
    if img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel
    
    hsv = rgb2hsv(img_array)
    # Use saturation channel for color-based detection
    saturation = hsv[:, :, 1]
    
    # Create mask based on saturation threshold
    mask = saturation > threshold
    
    # Clean up small noise
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    
    return mask


def mask_from_edge_detection(img: Image.Image) -> np.ndarray:
    """Create mask using edge detection and watershed segmentation."""
    img_array = np.asarray(img)
    if img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel
    
    # Convert to grayscale
    gray = rgb2gray(img_array)
    
    # Detect edges
    edges = sobel(gray)
    
    # Find local maxima for watershed seeds
    distance = ndimage.distance_transform_edt(edges < 0.1)
    local_max_coords = peak_local_max(distance, min_distance=20)
    local_max_mask = np.zeros_like(distance, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    
    # Watershed segmentation
    markers = ndimage.label(local_max_mask)[0]
    labels = watershed(-distance, markers, mask=edges < 0.1)
    
    # Create mask from watershed regions
    mask = labels > 0
    
    return mask


def mask_from_threshold(img: Image.Image) -> np.ndarray:
    """Create mask using Otsu thresholding."""
    img_array = np.asarray(img)
    if img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel
    
    # Convert to grayscale
    gray = rgb2gray(img_array)
    
    # Apply Otsu thresholding
    threshold_value = threshold_otsu(gray)
    mask = gray < threshold_value
    
    # Clean up noise
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    
    return mask


def create_mask(img: Image.Image, method: str, **kwargs) -> np.ndarray:
    """Create a mask using the specified detection method."""
    if method == "background_removal":
        # This would require the background removal step first
        raise ValueError("Background removal method requires the image to be processed first")
    elif method == "color":
        threshold = kwargs.get('color_threshold', 0.1)
        return mask_from_color_threshold(img, threshold)
    elif method == "edge":
        return mask_from_edge_detection(img)
    elif method == "threshold":
        return mask_from_threshold(img)
    else:
        raise ValueError(f"Unknown detection method: {method}")


def extract_and_save_objects(
    img: Image.Image,
    mask: np.ndarray,
    out_dir: Path,
    min_area: int = 300,
) -> None:
    """Detect connected components, crop, and save each object."""
    labeled = label(mask, connectivity=2)
    regions = [r for r in regionprops(labeled) if r.area >= min_area]

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, r in enumerate(regions, start=1):
        y0, x0, y1, x1 = r.bbox  # (min_row, min_col, max_row, max_col)
        cropped = img.crop((x0, y0, x1, y1))
        cropped.save(out_dir / f"object_{idx:02}.png")
    print(f"✅  Saved {len(regions)} objects to {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract objects using various detection methods")
    parser.add_argument("image", type=Path, help="Input image file")
    parser.add_argument(
        "--method",
        type=str,
        choices=["background_removal", "color", "edge", "threshold"],
        default="background_removal",
        help="Detection method to use",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=300,
        help="Minimum component area (pixels) to keep",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("objects"),
        help="Directory to save cropped objects",
    )
    parser.add_argument(
        "--save_bg_removed",
        action="store_true",
        help="Save the background-removed image (only for background_removal method)",
    )
    parser.add_argument(
        "--bg_removed_name",
        type=str,
        default=None,
        help="Name for the background-removed image (default: input_name_bg_removed.png)",
    )
    parser.add_argument(
        "--color_threshold",
        type=float,
        default=0.1,
        help="Color threshold for color-based detection (0.0-1.0)",
    )
    args = parser.parse_args()

    # Load the original image
    original_img = Image.open(args.image)
    
    if args.method == "background_removal":
        print("🔄  Removing background…")
        rgba_img = remove_background(args.image)
        
        # Save the background-removed image if requested
        if args.save_bg_removed:
            if args.bg_removed_name:
                bg_removed_path = args.out_dir / args.bg_removed_name
            else:
                # Generate default name: input_name_bg_removed.png
                input_stem = args.image.stem
                bg_removed_path = args.out_dir / f"{input_stem}_bg_removed.png"
            
            args.out_dir.mkdir(parents=True, exist_ok=True)
            rgba_img.save(bg_removed_path)
            print(f"💾  Saved background-removed image to {bg_removed_path.resolve()}")
        
        print("🔄  Generating mask from alpha channel…")
        mask = mask_from_alpha(rgba_img)
        working_img = rgba_img
        
    else:
        print(f"🔄  Generating mask using {args.method} detection…")
        mask = create_mask(original_img, args.method, color_threshold=args.color_threshold)
        working_img = original_img

    print("💾  Cropping and saving objects…")
    extract_and_save_objects(working_img, mask, args.out_dir, args.min_area)


if __name__ == "__main__":
    main()
