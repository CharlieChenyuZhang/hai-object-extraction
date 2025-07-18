#!/usr/bin/env python3
"""
Example usage of extract_objects.py with different detection methods.

This script demonstrates how to use the various object detection methods
without requiring background removal.
"""

import subprocess
import sys
from pathlib import Path


def run_extraction(image_path: str, method: str, min_area: int = 300, **kwargs):
    """Run the object extraction with specified parameters."""
    cmd = [
        sys.executable, "extract_objects.py",
        image_path,
        "--method", method,
        "--min_area", str(min_area),
        "--out_dir", f"objects_{method}"
    ]
    
    # Add method-specific parameters
    if method == "color" and "color_threshold" in kwargs:
        cmd.extend(["--color_threshold", str(kwargs["color_threshold"])])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!")
        print(result.stdout)
    else:
        print("❌ Error:")
        print(result.stderr)
    
    return result.returncode == 0


def main():
    """Demonstrate different detection methods."""
    # Check if an image was provided
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <image_path>")
        print("Example: python example_usage.py sample_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    print("🔍 Object Detection Methods Demo")
    print("=" * 50)
    
    # Test different methods
    methods = [
        ("color", {"color_threshold": 0.1}),
        ("edge", {}),
        ("threshold", {}),
    ]
    
    for method, params in methods:
        print(f"\n📸 Testing {method.upper()} detection method...")
        success = run_extraction(image_path, method, min_area=500, **params)
        
        if success:
            print(f"📁 Results saved in: objects_{method}/")
        else:
            print(f"❌ {method.upper()} method failed")
    
    print("\n🎉 Demo complete!")
    print("\nTo use background removal (original method):")
    print(f"  python extract_objects.py {image_path} --method background_removal")


if __name__ == "__main__":
    main() 