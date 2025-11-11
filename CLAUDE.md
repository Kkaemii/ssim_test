# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jupyter notebook-based project for analyzing and comparing anime/manga-style faces using a custom SSIM metric. The project implements **ERS-SSIM** (Existence-Relative-Structural SSIM), which extends traditional SSIM to better handle animated/illustrated character faces.

## Core Architecture

### ERS-SSIM Algorithm

The similarity metric is composed of three components multiplied together:

1. **Existence Term (E)**: Detects presence/absence of facial features (eyes, nose, mouth)
   - Uses MediaPipe Face Mesh for landmark detection
   - Compares which features exist in both images
   - Formula: \`l = (2*E1*E2 + C1) / (E1^2 + E2^2 + C1)\`

2. **Relative Geometry Term (R)**: Measures proportional distances between features
   - Eye distance ratio: \`d_eye_nose / d_eyes\`
   - Nose-mouth ratio: \`d_nose_mouth / d_eyes\`
   - Normalizes measurements relative to eye distance to be scale-invariant
   - Formula: \`c = (2*R + C2) / (R^2 + 1.0 + C2)\`

3. **Structural Term (S)**: Edge-based structural comparison
   - Uses Canny edge detection on grayscale faces
   - Applies scikit-image SSIM to edge maps with face feature masking
   - Formula: \`s = S\` (Edge SSIM used directly, no C3 constant)

Final score: \`SSIM = l * c * s\`

### Processing Pipeline

1. **Face Detection**: Two models available
   - **YOLOv8** (default, recommended): \`Fuyucchi/yolov8_animeface\` from Hugging Face
     - Higher accuracy for anime/manga faces
     - Confidence threshold: 0.25 (adjustable)
     - Auto-downloads on first use via \`ultralyticsplus\` package
     - Fallback to Cascade if detection fails
   - **Cascade**: \`lbpcascade_animeface.xml\` (anime-optimized Haar Cascade)
     - Located in root directory
     - Falls back to more permissive parameters if initial detection fails
   - Both add 10% margin around detected face boxes

2. **Landmark Detection**: MediaPipe Face Mesh with 468 landmarks
   - Key indices: left_eye (33-133), right_eye (362-463), nose (1), mouth (61-291)

3. **Score Calculation**: Combines E, R, S terms with safety checks for division by zero

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
# Using YOLOv8 (default, recommended)
process_images('images/img1-1.png', 'images/img1-2.png', method='yolo')

# Using Cascade (fallback)
process_images('images/img1-1.png', 'images/img1-2.png', method='cascade')
\`\`\`

## Key Dependencies

- OpenCV (\`cv2\`): Image processing and anime face detection
- **ultralyticsplus**: Hugging Face integration for YOLOv8 models
- **ultralytics**: YOLOv8 inference engine
- **PyTorch** (\`torch\`, \`torchvision\`): Required for YOLOv8
- MediaPipe (\`mediapipe\`): Facial landmark detection (468-point mesh)
- scikit-image (\`skimage.metrics\`): SSIM calculation for edge maps
- NumPy/Matplotlib: Numerical computation and visualization
- PIL: Image loading support

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

### Selecting Face Detection Model

\`\`\`python
# Use YOLOv8 (higher accuracy, recommended)
process_images('img1.png', 'img2.png', method='yolo')

# Use Cascade (faster, lower accuracy)
process_images('img1.png', 'img2.png', method='cascade')
\`\`\`

### Adjusting Face Detection Sensitivity

**For YOLOv8:**
\`\`\`python
# In detect_anime_faces_yolo() or when calling detect_anime_faces()
detect_anime_faces(img, method='yolo', conf_threshold=0.3)  # Higher = more strict
\`\`\`

**For Cascade:**
\`\`\`python
# In detect_anime_faces() function:
detect_anime_faces(img, method='cascade', scale_factor=1.1, min_neighbors=5)
# - Decrease scaleFactor (default 1.1) for more detections
# - Decrease minNeighbors (default 5) for more permissive detection
\`\`\`

### Tuning SSIM Components

In \`ers_ssim()\` function at face_comparison-Copy1.ipynb:
- \`C1\` (default 1e-6): Existence term stability constant
- \`C2\` (default 0.5): Geometry term stability constant
- Structural term uses Edge SSIM directly (no C3 constant)

The structural term applies MediaPipe landmark-based masking to focus edge detection on facial features only (face oval, eyes, nose, lips).

### Modifying Edge Detection

In \`structural_term()\` function:
- \`cv2.Canny(g1_resized, 80, 150)\`: Adjust thresholds (80, 150) for edge sensitivity
- Lower thresholds = more edges detected

## Architecture Notes

- All processing happens in-memory; no intermediate files saved
- Global variables \`face_images\` and \`original_images\` store extracted faces across cells
- Korean font support attempted for matplotlib titles (falls back to DejaVu Sans)
- Face margin (10%) ensures full feature capture even with imperfect detection
- YOLOv8 model uses lazy loading (only loaded when first used)
- Automatic fallback from YOLOv8 to Cascade if detection fails
