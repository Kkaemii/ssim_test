# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jupyter notebook-based project for analyzing and comparing anime/manga-style character faces using **standard SSIM** (Structural Similarity Index). The project focuses on measuring character consistency across generated images.

## Core Architecture

### SSIM-Based Similarity Measurement

The project uses two complementary metrics:

1. **Standard SSIM** (Primary metric, 70% weight)
   - Measures pixel-level structural similarity on grayscale face images
   - Uses scikit-image's `structural_similarity` function
   - Compares luminance, contrast, and structure
   - Range: 0.0 (completely different) to 1.0 (identical)

2. **Edge SSIM** (Optional secondary metric, 30% weight)
   - Measures similarity of facial contours and line work
   - Uses Canny edge detection followed by SSIM
   - Focuses on outline consistency rather than pixel details
   - Useful for comparing illustration styles

**Final Score**: `(Standard SSIM × 0.7) + (Edge SSIM × 0.3)`

### Processing Pipeline

1. **Face Detection**: YOLOv8 anime face detector
   - **Model**: `Fuyucchi/yolov8_animeface` (local file: `yolov8x6_animeface.pt`)
     - YOLOv8x6 architecture trained for anime/manga faces
     - mAP50: 0.953, Precision: 0.956
   - **Parameters**:
     - Confidence threshold: 0.25 (adjustable)
     - Margin: 0% by default (tight crop on face only)
     - Selects largest face if multiple detected
   - **Note**: Cascade model (`lbpcascade_animeface.xml`) available but not currently used

2. **Face Comparison**: Standard SSIM calculation
   - Resizes faces to matching dimensions
   - Converts to grayscale
   - Calculates SSIM using scikit-image
   - Optionally calculates Edge SSIM for contour comparison

3. **No Landmark Detection**:
   - Previous versions used MediaPipe for landmark detection
   - Removed due to poor performance on anime faces (trained for real faces)
   - Standard SSIM works directly on face crops without landmarks

## Running the Notebook

### Start Jupyter

\`\`\`bash
jupyter notebook face_comparison-Copy1.ipynb
\`\`\`

### Running All Cells

In Jupyter interface: Cell → Run All

Or use nbconvert:
\`\`\`bash
jupyter nbconvert --to notebook --execute face_comparison-Copy1.ipynb --output executed.ipynb
\`\`\`

### Running Specific Analysis

The notebook auto-executes \`process_images()\` at the end of cell 2. To analyze different images:

\`\`\`python
# Basic usage (0% margin, tight face crop)
process_images('images/img1.png', 'images/img2.png')

# With margin (include some background)
process_images('images/img1.png', 'images/img2.png', margin_percent=0.1)

# Adjust confidence threshold
process_images('images/img1.png', 'images/img2.png', conf_threshold=0.3)
\`\`\`

After face extraction, run comparison:

\`\`\`python
# Compare with both standard and edge SSIM
result = compare_faces(face_images[0], face_images[1], use_edge_ssim=True, edge_weight=0.3)

# Use only standard SSIM
result = compare_faces(face_images[0], face_images[1], use_edge_ssim=False)
\`\`\`

## Key Dependencies

- OpenCV (\`cv2\`): Image processing and Canny edge detection
- **ultralytics**: YOLOv8 inference engine for anime face detection
- **PyTorch** (\`torch\`, \`torchvision\`): Required for YOLOv8
- scikit-image (\`skimage.metrics\`): Standard SSIM calculation
- NumPy/Matplotlib: Numerical computation and visualization
- PIL: Image loading support
- pandas: Results table generation (for batch comparison)

**Removed Dependencies** (from previous version):
- MediaPipe: No longer used (poor performance on anime faces)
- ultralyticsplus: Not needed (using local model file)

## Model Files

- **YOLOv8 Model**: \`Fuyucchi/yolov8_animeface\` (auto-downloaded from Hugging Face)
  - YOLOv8x6 architecture trained for 300 epochs
  - Input size: 1280x1280px
  - mAP50: 0.953, Precision: 0.956
- \`lbpcascade_animeface.xml\`: Pre-trained Haar Cascade for anime face detection (fallback)
- \`res10_300x300_ssd_iter_140000.caffemodel\`: Caffe SSD model (appears unused in current code)

## Test Images

Located in \`images/\` directory with naming patterns:
- \`img{n}-1.png\` / \`img{n}-2.png\`: Paired comparison images
- \`moondoors_*.png\`: Generated character images from API

## Common Development Tasks

### Adjusting Face Detection

\`\`\`python
# Adjust confidence threshold (higher = stricter)
process_images('img1.png', 'img2.png', conf_threshold=0.3)

# Add margin around face (0.0 = tight crop, 0.1 = 10% padding)
process_images('img1.png', 'img2.png', margin_percent=0.1)
\`\`\`

### Customizing SSIM Calculation

\`\`\`python
# Use only standard SSIM (no edge comparison)
result = compare_faces(face1, face2, use_edge_ssim=False)

# Adjust edge SSIM weight (default 0.3 = 30%)
result = compare_faces(face1, face2, use_edge_ssim=True, edge_weight=0.5)
\`\`\`

### Modifying Edge Detection

In \`calculate_edge_ssim()\` function:
\`\`\`python
# Adjust Canny thresholds (default: 80, 150)
edge_ssim, edge1, edge2 = calculate_edge_ssim(face1, face2, canny_low=50, canny_high=100)
# - Lower thresholds = more edges detected
# - Higher thresholds = only strong edges
\`\`\`

### Batch Processing

See Cell 7 for batch comparison of multiple images:
\`\`\`python
# Edit file_list to add your images
file_list = ['images/img1.png', 'images/img2.png', ...]

# Results are saved in pandas DataFrame for easy analysis
df = pd.DataFrame(results)
df_sorted = df.sort_values('최종점수', ascending=False)
\`\`\`

## Architecture Notes

- All processing happens in-memory; no intermediate files saved
- Global variables \`face_images\` and \`original_images\` store extracted faces across cells
- Korean font support attempted for matplotlib titles (falls back to DejaVu Sans)
- Face margin configurable (default 0% for tight crop)
- YOLOv8 model uses lazy loading (only loaded when first used)
- No fallback detection method (Cascade model available but not integrated)

## Limitations

- **Angle sensitivity**: SSIM compares pixel patterns, so faces at different angles will score lower even if the same character
- **Expression changes**: Different facial expressions affect similarity scores
- **Style variations**: Works best when comparing images with consistent art style
- **No feature matching**: Unlike face recognition models, this doesn't extract identity features

For detecting the "same character" across different poses/angles, consider using:
- Anime face recognition models (e.g., AnimeFace feature extractors)
- Multi-view face alignment before SSIM
- Higher-level feature embeddings rather than pixel-level comparison

## Version History

**Current Version**: Standard SSIM-based comparison
- Removed ERS-SSIM (custom metric with E, R, S components)
- Removed MediaPipe landmark detection (poor anime face support)
- Simplified to standard SSIM + optional Edge SSIM
- 50%+ code reduction for better maintainability

**Previous Version**: ERS-SSIM (archived in older commits)
- Used MediaPipe for landmark detection
- Custom Existence, Relative Geometry, and Structural terms
- More complex but failed frequently on anime faces
