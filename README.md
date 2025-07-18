# Object Extraction Tool

A Python tool for extracting objects from images using various detection methods, with or without background removal.

## Features

- **Multiple Detection Methods**: Choose from 4 different approaches to detect objects
- **Background Removal**: Original method using AI-powered background removal
- **Color-based Detection**: Segment objects based on color saturation
- **Edge-based Detection**: Use edge detection and watershed segmentation
- **Threshold-based Detection**: Simple grayscale thresholding with Otsu's method
- **Flexible Output**: Save individual objects as PNG files

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd hai-object-extraction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Use background removal (original method)
python extract_objects.py input_image.jpg

# Use color-based detection
python extract_objects.py input_image.jpg --method color

# Use edge-based detection
python extract_objects.py input_image.jpg --method edge

# Use threshold-based detection
python extract_objects.py input_image.jpg --method threshold
```

### Command Line Options

- `--method`: Detection method to use
  - `background_removal` (default): AI-powered background removal
  - `color`: Color-based segmentation using HSV saturation
  - `edge`: Edge detection with watershed segmentation
  - `threshold`: Otsu thresholding on grayscale
- `--min_area`: Minimum object area in pixels (default: 300)
- `--out_dir`: Output directory for extracted objects (default: "objects")
- `--color_threshold`: Color threshold for color method (0.0-1.0, default: 0.1)
- `--save_bg_removed`: Save background-removed image (background_removal method only)

### Examples

```bash
# Extract objects with minimum area of 500 pixels using color detection
python extract_objects.py images/test1.jpg --method color --min_area 500

# Use edge detection and save to custom directory
python extract_objects.py images/test1.jpg --method edge --out_dir my_objects

# Adjust color threshold for better detection
python extract_objects.py images/test1.jpg --method color --color_threshold 0.2

# Use background removal and save the processed image
python extract_objects.py images/test1.jpg --method background_removal --save_bg_removed
```

## Detection Methods

### 1. Background Removal (Original)

- Uses AI-powered background removal via `rembg`
- Creates transparent background
- Detects objects from alpha channel
- Best for images with complex backgrounds

### 2. Color-based Detection

- Converts image to HSV color space
- Uses saturation channel for segmentation
- Good for colorful objects on neutral backgrounds
- Adjustable threshold parameter

### 3. Edge-based Detection

- Uses Sobel edge detection
- Applies watershed segmentation
- Good for objects with clear boundaries
- Works well with complex shapes

### 4. Threshold-based Detection

- Converts to grayscale
- Uses Otsu's automatic thresholding
- Simple and fast
- Good for high-contrast images

## Demo Script

Run the demo script to test all methods on your image:

```bash
python example_usage.py your_image.jpg
```

This will create separate output directories for each method:

- `objects_color/`
- `objects_edge/`
- `objects_threshold/`

## Output

The tool creates:

- Individual PNG files for each detected object
- Files named `object_01.png`, `object_02.png`, etc.
- Background-removed image (if requested)

## Dependencies

- `pillow`: Image processing
- `rembg`: AI background removal
- `numpy`: Numerical operations
- `scikit-image`: Image analysis and processing
- `scipy`: Scientific computing (morphological operations)
- `onnxruntime`: Required by rembg

## Tips

1. **Choose the right method**:

   - Use `background_removal` for complex backgrounds
   - Use `color` for colorful objects on neutral backgrounds
   - Use `edge` for objects with clear boundaries
   - Use `threshold` for high-contrast images

2. **Adjust parameters**:

   - Increase `min_area` to filter out small noise
   - Adjust `color_threshold` for better color segmentation
   - Try different methods on the same image

3. **Image quality**:
   - Higher resolution images work better
   - Good lighting improves detection accuracy
   - Clear object boundaries help all methods

## Troubleshooting

- **No objects detected**: Try reducing `min_area` or adjusting `color_threshold`
- **Too many small objects**: Increase `min_area` or use morphological operations
- **Poor color detection**: Adjust `color_threshold` or try a different method
- **Memory issues**: Use smaller images or increase system memory
