# Project Documentation

## Detailed Project Descriptions

### 1. White Blood Cell Classification System

#### Overview
This project implements a complete pipeline for classifying white blood cells into four main categories: Lymphocytes, Monocytes, Neutrophils, and Eosinophils, using traditional computer vision techniques without machine learning.

#### Technical Implementation

**Image Preprocessing:**
- HSV color space conversion for better color-based segmentation
- Gaussian filtering for noise reduction
- Dynamic thresholding using mean and standard deviation
- Morphological operations for shape refinement

**Segmentation Algorithm:**
```python
def connectivit_8(image, upp_limit, low_limit=0):
    # 8-connectivity component labeling with union-find optimization
    # Supports intensity-based thresholding
    # Returns labeled regions for analysis
```

**Feature Extraction:**
- **Shape Features**: Area, perimeter, eccentricity, solidity, extent, circularity
- **Color Features**: Mean hue, saturation, and value statistics
- **Texture Features**: Local Binary Pattern (LBP) analysis
- **Morphological Features**: Contour-based lobe counting

**Classification Logic:**
- Rule-based classification using feature thresholds
- Multi-criteria decision making
- Outlier removal using IQR method

#### Performance Metrics
- Dice Coefficient for segmentation accuracy
- Pixel-wise accuracy measurement
- Confusion matrix analysis
- Per-class precision and recall

### 2. Lane Detection and Tracking System

#### Overview
Real-time lane detection system capable of processing video streams and identifying lane boundaries, curvature, and potential obstacles.

#### Technical Implementation

**Preprocessing Pipeline:**
- Gaussian blur for noise reduction
- Color space conversion (RGB to HSV/HLS)
- Edge detection using Canny algorithm
- Region of Interest (ROI) masking

**Lane Detection Algorithm:**
```python
def detect_lanes(frame):
    # 1. Preprocess frame
    # 2. Apply edge detection
    # 3. Hough line transformation
    # 4. Line filtering and grouping
    # 5. Lane boundary fitting
    # 6. Curvature calculation
```

**Advanced Features:**
- **Solid vs Dashed Lane Detection**: Pattern analysis of detected lines
- **Perspective Transformation**: Bird's eye view for accurate measurements
- **Lane Departure Warning**: Real-time monitoring of vehicle position
- **Obstacle Detection**: Background subtraction for moving objects

**State Machine Implementation:**
- Lane tracking with stability thresholds
- Adaptive parameter tuning based on detection confidence
- Frame-to-frame consistency checking

### 3. Texture Analysis using GLCM

#### Overview
Comprehensive texture analysis implementation using Gray-Level Co-occurrence Matrix (GLCM) for feature extraction and pattern recognition.

#### Technical Implementation

**GLCM Calculation:**
```python
def compute_glcm(image, distance=1, angle=0):
    # Compute co-occurrence matrix
    # Support for multiple distances and angles
    # Symmetric and normalized options
```

**Texture Features:**
- **Energy/Uniformity**: Measures texture uniformity
- **Contrast**: Local intensity variation
- **Correlation**: Linear dependency of gray levels
- **Entropy**: Randomness measure
- **Homogeneity**: Closeness of distribution to diagonal

**Spectral Analysis:**
- Fourier Transform-based frequency analysis
- Radial and angular feature profiles S(r) and S(θ)
- Dominant frequency and orientation detection

### 4. Medical Image Segmentation

#### Overview
Advanced segmentation techniques for medical images, specifically focused on nucleus and cytoplasm separation in cell images.

#### Technical Implementation

**Multi-level Thresholding:**
- Otsu's method implementation
- Adaptive thresholding based on local statistics
- Hysteresis thresholding for edge connectivity

**Segmentation Validation:**
```python
def dice_coefficient(pred_mask, true_mask):
    # Dice similarity coefficient calculation
    # Handles multi-class segmentation
    # Returns class-wise and overall scores
```

**Post-processing:**
- Morphological operations for mask refinement
- Hole filling algorithms
- Connected component filtering

## Laboratory Exercise Details

### Lab 1-2: Fundamentals
- **Image I/O Operations**: Reading, writing, and displaying images
- **Color Space Conversions**: RGB, HSV, LAB transformations
- **Basic Array Operations**: NumPy integration with OpenCV

### Lab 3: Histogram Analysis
- **Histogram Calculation**: Manual and built-in implementations
- **Histogram Equalization**: Global and adaptive methods
- **Statistical Analysis**: Mean, variance, skewness, kurtosis

### Lab 4: Spatial Filtering
- **Convolution Implementation**: Manual convolution operations
- **Filter Design**: Gaussian, Laplacian, high-pass, low-pass filters
- **Noise Reduction**: Various denoising techniques

### Lab 5: Frequency Domain
- **FFT Implementation**: 2D Fourier Transform operations
- **Filter Design**: Ideal, Butterworth, Gaussian filters
- **Image Enhancement**: Frequency domain techniques

### Lab 6: Edge Detection
- **Gradient Operators**: Sobel, Prewitt, Roberts cross-gradient
- **Canny Edge Detector**: Complete implementation with hysteresis
- **Edge Linking**: Contour following and gap filling

### Lab 9: Morphological Operations
- **Structuring Elements**: Various shapes and sizes
- **Basic Operations**: Erosion, dilation, opening, closing
- **Advanced Operations**: Hit-or-miss, skeletonization

### Lab 10: Feature Extraction
- **HOG Descriptors**: Histogram of Oriented Gradients
- **Shape Descriptors**: Moments, Fourier descriptors
- **Texture Features**: LBP, GLCM-based features

### Lab 12: Advanced Texture Analysis
- **GLCM Properties**: Complete texture feature set
- **Spectral Analysis**: Frequency domain texture analysis
- **Pattern Recognition**: Texture-based classification

### Lab 13: Classification
- **k-NN Implementation**: Manual k-nearest neighbors
- **Feature Selection**: Dimensionality reduction techniques
- **Performance Evaluation**: Cross-validation, metrics calculation

## Code Architecture

### Utility Functions
```python
# Image processing utilities
def image_read(path, crop=True, crop_coords=None)
def show_image(img, title="Image", scale=1.0)
def cal_histogram(img, bins=256)

# Advanced processing
def gauss_filter(img, sigma=1.0)
def thresholding(img, method='otsu')
def connected_components(binary_img, connectivity=8)
```

### Feature Extraction Pipeline
```python
class FeatureExtractor:
    def __init__(self):
        self.features = {}
    
    def extract_shape_features(self, mask)
    def extract_color_features(self, img, mask)
    def extract_texture_features(self, img, mask)
    def get_feature_vector(self)
```

### Performance Monitoring
```python
class PerformanceTracker:
    def __init__(self):
        self.metrics = {}
    
    def calculate_dice(self, pred, truth)
    def calculate_accuracy(self, pred, truth)
    def generate_confusion_matrix(self, pred, truth)
```

## Dataset Information

### WBC Dataset
- **Classes**: 4 (Lymphocyte, Monocyte, Neutrophil, Eosinophil)
- **Images**: 250+ annotated cell images
- **Resolution**: Variable (typically 640x480)
- **Format**: JPEG, PNG
- **Annotations**: Ground truth masks for nucleus and cytoplasm

### Lane Detection Dataset
- **Videos**: Multiple driving scenarios
- **Conditions**: Day/night, different weather conditions
- **Resolution**: 1080p, 720p
- **Duration**: Various lengths (30 seconds to 5 minutes)

## Research Applications

### Medical Imaging
- **Hematology**: Automated blood cell counting
- **Pathology**: Cell abnormality detection
- **Diagnostics**: Early disease detection

### Autonomous Vehicles
- **ADAS Systems**: Advanced Driver Assistance Systems
- **Navigation**: Lane-keeping assistance
- **Safety**: Collision avoidance systems

### Industrial Applications
- **Quality Control**: Surface defect detection
- **Manufacturing**: Automated inspection systems
- **Robotics**: Vision-guided manipulation