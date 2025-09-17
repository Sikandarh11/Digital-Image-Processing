# 🖼️ Digital Image Processing Laboratory

A comprehensive collection of Digital Image Processing projects, assignments, and laboratory exercises implemented in Python using OpenCV, NumPy, and various computer vision techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Major Projects](#major-projects)
- [Laboratory Exercises](#laboratory-exercises)
- [Installation & Setup](#installation--setup)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Key Technologies](#key-technologies)
- [Contributors](#contributors)

## 🔍 Overview

This repository contains a collection of Digital Image Processing projects developed as part of academic coursework. The projects cover a wide range of computer vision and image analysis techniques, from basic image manipulation to advanced feature extraction and classification algorithms.

### Key Focus Areas:
- **Medical Image Analysis** - Cell classification and nucleus segmentation
- **Computer Vision** - Lane detection and object tracking
- **Feature Extraction** - Texture analysis and statistical descriptors
- **Image Enhancement** - Filtering, noise reduction, and preprocessing
- **Pattern Recognition** - Classification without machine learning

## 📁 Repository Structure

```
Digital-Image-Processing/
├── 1. Classification of Nucleus without ML & DL/
│   └── White Blood Cell classification using traditional CV methods
├── 2. Nucleus + Cytoplasm_Mask/
│   └── Cell segmentation and masking techniques
├── DIP Lab/
│   └── Projects/
│       ├── Assignment/           # Course assignments
│       ├── Assignment 2/         # Advanced WBC classification
│       ├── Dip Project/          # Lane detection system
│       ├── LAB/                  # Weekly lab exercises (1-13)
│       ├── Lab 3-13/             # Specific lab implementations
│       └── Open Lab with Instructions/
└── README.md
```

## 🎯 Major Projects

### 1. 🩸 White Blood Cell Classification
**Location:** `1. Classification of Nucleus without ML & DL/`

A comprehensive system for classifying white blood cells using traditional computer vision techniques without machine learning.

**Features:**
- HSV color space analysis for nucleus segmentation
- 8-connectivity component labeling
- Handcrafted feature extraction (area, perimeter, circularity, etc.)
- Multi-class WBC classification (Neutrophil, Eosinophil, Monocyte, Lymphocyte)
- Statistical analysis and accuracy measurement

**Key Techniques:**
- Gaussian filtering for noise reduction
- Custom thresholding algorithms
- Connected component analysis
- Texture feature extraction using LBP (Local Binary Patterns)

### 2. 🧬 Nucleus and Cytoplasm Segmentation
**Location:** `2. Nucleus + Cytoplasm_Mask/`

Advanced cell segmentation techniques for medical image analysis.

**Features:**
- Nucleus boundary detection
- Cytoplasm region identification
- Mask generation and validation
- Accuracy computation against ground truth

### 3. 🛣️ Lane Detection System
**Location:** `DIP Lab/Projects/Dip Project/`

Real-time lane detection system for autonomous vehicle applications.

**Features:**
- Video processing and frame analysis
- Hough line transformation for lane detection
- Dashed vs solid lane classification
- State machine for lane tracking
- Real-time FPS calculation
- Background subtraction for obstacle detection

**Technical Implementation:**
- Canny edge detection
- Region of interest masking
- Line slope analysis
- Temporal stability filtering

### 4. 📊 Advanced WBC Analysis
**Location:** `DIP Lab/Projects/Assignment 2/`

Enhanced white blood cell analysis with comprehensive feature extraction.

**Features:**
- Multi-dimensional feature space analysis
- HSV histogram computation
- Statistical feature calculation
- Dataset preprocessing and outlier removal
- Comparative analysis across cell types

## 🔬 Laboratory Exercises

The repository includes 13+ laboratory exercises covering fundamental DIP concepts:

### Lab Topics Include:
- **Lab 1-2:** Basic image operations and matrix manipulations
- **Lab 3:** Histogram analysis and equalization
- **Lab 4:** Spatial filtering and convolution
- **Lab 5:** Morphological operations
- **Lab 6:** Edge detection (Canny, Sobel)
- **Lab 9:** Advanced filtering techniques
- **Lab 10:** Feature detection and matching
- **Lab 11:** Image segmentation
- **Lab 12:** Texture analysis and GLCM features
- **Lab 13:** Pattern recognition and classification

### Special Projects:
- **Emotion Detection** - Facial expression analysis
- **HOG Features** - Histogram of Oriented Gradients implementation
- **Spectral Analysis** - Frequency domain feature extraction

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.7+
pip package manager
```

### Required Libraries
```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-image
pip install scikit-learn
pip install seaborn
```

### Optional Dependencies
```bash
pip install jupyter          # For running notebooks
pip install ipykernel       # Jupyter kernel support
pip install plotly          # Interactive visualizations
```

## 📦 Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| OpenCV | Computer vision operations | >= 4.5.0 |
| NumPy | Numerical computations | >= 1.19.0 |
| Matplotlib | Plotting and visualization | >= 3.3.0 |
| Pandas | Data manipulation | >= 1.2.0 |
| Scikit-image | Image processing algorithms | >= 0.18.0 |
| Scikit-learn | Machine learning utilities | >= 0.24.0 |

## 🚀 Usage

### Running Jupyter Notebooks
```bash
# Navigate to project directory
cd Digital-Image-Processing

# Start Jupyter
jupyter notebook

# Open desired notebook and run cells
```

### Running Individual Scripts
```bash
# For Python scripts
python DIP\ Lab/Projects/Open\ Lab\ with\ Instructions/lab10taskhog.py
```

### Example: WBC Classification
```python
# Load and preprocess image
img_hsv = image_read_hsv("path/to/cell_image.jpg")

# Extract features
features = extract_additional_features(img_hsv, binary_mask)

# Classify cell type
cell_type = classify_wbc(features)
print(f"Detected cell type: {cell_type}")
```

### Example: Lane Detection
```python
# Initialize video capture
cap = cv2.VideoCapture("road_video.mp4")

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    lane_lines = detect_lanes(frame)
    annotated_frame = draw_lanes(frame, lane_lines)
    cv2.imshow("Lane Detection", annotated_frame)
```

## 🔧 Key Technologies

### Computer Vision Techniques
- **Edge Detection:** Canny, Sobel, Laplacian
- **Feature Extraction:** SIFT, SURF, HOG, LBP
- **Morphological Operations:** Erosion, dilation, opening, closing
- **Filtering:** Gaussian, median, bilateral
- **Segmentation:** Watershed, region growing, thresholding

### Image Analysis Methods
- **Color Space Conversion:** RGB, HSV, LAB, Grayscale
- **Histogram Analysis:** Equalization, matching, statistics
- **Texture Analysis:** GLCM, statistical moments
- **Geometric Transformations:** Rotation, scaling, perspective

### Mathematical Foundations
- **Linear Algebra:** Matrix operations, eigenvectors
- **Statistics:** Moments, entropy, uniformity
- **Signal Processing:** FFT, filtering, convolution
- **Optimization:** Feature selection, thresholding

## 👥 Contributors

- **Sikandar** - Primary developer and researcher
- **Course Instructor** - Academic guidance and requirements
- **Lab Partners** - Collaborative development and testing

## 📈 Project Statistics

- **Total Notebooks:** 42 Jupyter notebooks
- **Python Scripts:** 2 standalone scripts
- **Lab Exercises:** 13+ comprehensive labs
- **Major Projects:** 4 complete implementations
- **Code Coverage:** Image processing, computer vision, medical imaging

## 🎓 Academic Context

This repository represents coursework for a Digital Image Processing course, demonstrating practical implementation of theoretical concepts. Each project includes:

- Detailed code documentation
- Algorithm explanations
- Performance analysis
- Visual results and comparisons
- Academic references and methodology

## 📝 Notes

- All implementations use traditional computer vision methods
- No deep learning or neural networks used (as per course requirements)
- Focus on understanding fundamental image processing concepts
- Extensive use of OpenCV and NumPy for efficient computation
- Real-world applications in medical imaging and autonomous systems

---

*For questions, issues, or contributions, please refer to the individual project notebooks or contact the repository maintainer.*