# Digital Image Processing Repository 🖼️

A comprehensive collection of Digital Image Processing (DIP) projects, assignments, and laboratory exercises implemented in Python using OpenCV, NumPy, and other computer vision libraries.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Key Projects](#key-projects)
- [Laboratory Exercises](#laboratory-exercises)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Highlights](#project-highlights)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains a complete set of digital image processing implementations covering fundamental to advanced topics in computer vision and image analysis. The projects demonstrate practical applications of image processing techniques without relying on machine learning or deep learning approaches, focusing on traditional computer vision methods.

### Main Focus Areas:
- **Medical Image Analysis**: White Blood Cell (WBC) classification and segmentation
- **Computer Vision Applications**: Lane detection, emotion recognition
- **Image Processing Fundamentals**: Filtering, edge detection, morphological operations
- **Texture Analysis**: GLCM-based feature extraction
- **Image Enhancement**: Noise reduction, histogram equalization
- **Pattern Recognition**: Shape analysis and feature extraction

## 📁 Repository Structure

```
Digital-Image-Processing/
├── 1. Classification of Nucleus without ML & DL/
│   └── Sikandar_Dip_ass2.ipynb          # WBC classification implementation
├── 2. Nucleus + Cytoplasm_Mask/
│   └── Mask_DIP.ipynb                   # Cell segmentation and masking
├── DIP Lab/
│   └── Projects/
│       ├── Assignment/                   # Main assignments
│       ├── Assignment 2/                 # Advanced assignments
│       ├── Dip Project/                  # Lane detection project
│       ├── Lab 3/ to Lab 13/            # Individual lab exercises
│       ├── LAB/                         # Additional lab implementations
│       ├── Open Lab with Instructions/   # Guided lab exercises
│       └── siku-detecting-emotions.ipynb # Emotion detection project
└── README.md
```

## 🚀 Key Projects

### 1. White Blood Cell Classification 🩸
**Location**: `1. Classification of Nucleus without ML & DL/`
- **Objective**: Classify different types of white blood cells without using ML/DL
- **Techniques**: 
  - Custom thresholding algorithms
  - 8-connectivity component labeling
  - Morphological operations
  - Shape and texture feature extraction
  - HSV color space analysis
- **Features Extracted**: Area, perimeter, eccentricity, solidity, circularity, hue statistics
- **Output**: Classification into Lymphocyte, Monocyte, Neutrophil, and Eosinophil

### 2. Lane Detection System 🛣️
**Location**: `DIP Lab/Projects/Dip Project/`
- **Objective**: Real-time lane detection and tracking from video streams
- **Techniques**:
  - Gaussian blur and edge detection
  - Hough line transformation
  - Region of interest (ROI) masking
  - Perspective transformation
  - Lane curvature calculation
- **Features**: 
  - Solid vs dashed lane detection
  - Lane departure warnings
  - Real-time FPS calculation
  - Obstacle detection capabilities

### 3. Emotion Detection from EEG Data 🧠
**Location**: `DIP Lab/Projects/siku-detecting-emotions.ipynb`
- **Objective**: Predict emotional states from EEG signals during movie watching
- **Techniques**: Signal processing and feature extraction from EEG data
- **Application**: Brain-computer interface and affective computing

### 4. Nucleus and Cytoplasm Segmentation 🔬
**Location**: `2. Nucleus + Cytoplasm_Mask/`
- **Objective**: Accurate segmentation of cell components
- **Techniques**:
  - Multi-level thresholding
  - Connected component analysis
  - Mask generation and validation
- **Evaluation**: Dice coefficient and pixel accuracy metrics

## 🧪 Laboratory Exercises

The repository includes 13+ comprehensive lab exercises covering:

### Basic Image Processing (Labs 1-6)
- **Lab 1-2**: Image reading, display, and basic operations
- **Lab 3**: Histogram analysis and equalization
- **Lab 4**: Spatial filtering and convolution
- **Lab 5**: Frequency domain filtering
- **Lab 6**: Edge detection (Canny, Sobel, Prewitt)

### Advanced Topics (Labs 7-13)
- **Lab 9**: Morphological operations (erosion, dilation, opening, closing)
- **Lab 10**: Feature extraction and HOG descriptors
- **Lab 11**: Image restoration and noise removal
- **Lab 12**: Texture analysis using GLCM (Gray-Level Co-occurrence Matrix)
- **Lab 13**: Pattern recognition and classification

## 🛠️ Technologies Used

### Core Libraries
- **OpenCV**: Computer vision and image processing operations
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Data visualization and plotting
- **Pandas**: Data manipulation and analysis
- **scikit-image**: Advanced image processing algorithms

### Specialized Libraries
- **scikit-learn**: Machine learning utilities for evaluation metrics
- **seaborn**: Statistical data visualization
- **collections**: Data structures for algorithms

### Development Environment
- **Jupyter Notebook**: Interactive development and visualization
- **Python 3.7+**: Primary programming language

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/Sikandarh11/Digital-Image-Processing.git
cd Digital-Image-Processing
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv dip_env
source dip_env/bin/activate  # On Windows: dip_env\Scripts\activate
```

3. **Install required packages**:
```bash
pip install opencv-python numpy matplotlib pandas scikit-image scikit-learn seaborn jupyter
```

4. **For specific advanced features**, you may need:
```bash
pip install notebook ipywidgets
```

### Jupyter Notebook Setup
```bash
jupyter notebook
```
Navigate to the desired project folder and open the `.ipynb` files.

## 📚 Usage

### Running Individual Projects

#### WBC Classification Example:
```bash
cd "1. Classification of Nucleus without ML & DL"
jupyter notebook Sikandar_Dip_ass2.ipynb
```

#### Lane Detection Example:
```bash
cd "DIP Lab/Projects/Dip Project"
jupyter notebook proj.ipynb
```

### Key Functions and Modules

#### Image Processing Utilities:
```python
# Basic image operations
image_read(path, crop=True)          # Read and optionally crop images
show_image(img, scale=1.0)           # Display images with scaling
cal_histogram(img)                   # Calculate RGB histograms

# Advanced processing
connectivit_8(image, upp_limit, low_limit=0)  # 8-connectivity labeling
gauss_filter(img, sigma)             # Gaussian filtering
thresholding(img, threshold)         # Custom thresholding
```

#### Feature Extraction:
```python
# Shape and texture features
get_region_props(mask)               # Basic shape descriptors
extract_additional_features(img, mask)  # Advanced feature set
get_lbp_mean(img, mask)             # Local Binary Pattern features
```

## 🌟 Project Highlights

### Advanced Algorithms Implemented:
- **Custom Connected Component Labeling**: 8-connectivity algorithm with path compression
- **Multi-scale Gaussian Filtering**: Manual implementation for noise reduction
- **Dynamic Thresholding**: Adaptive thresholding using statistical measures
- **Perspective Transformation**: For lane detection and bird's eye view
- **GLCM Texture Analysis**: Complete texture feature extraction pipeline

### Performance Metrics:
- **WBC Classification**: Achieves high accuracy using handcrafted features
- **Lane Detection**: Real-time processing with FPS monitoring
- **Segmentation**: Evaluated using Dice coefficient and pixel accuracy

### Visualization Features:
- Comprehensive plotting functions for results analysis
- Interactive Jupyter notebook environment
- Before/after comparison visualizations
- Feature distribution analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines:
1. Follow PEP 8 style guidelines
2. Add comprehensive comments to your code
3. Include test cases for new features
4. Update documentation as needed

## 📝 License

This project is available under the MIT License. See the LICENSE file for more details.

## 📞 Contact

For questions, suggestions, or collaborations, please reach out through GitHub issues or contact the repository owner.

## 🙏 Acknowledgments

- Computer Vision community for open-source libraries
- Academic resources and research papers that inspired these implementations
- Contributors and reviewers who helped improve the codebase

---

**Note**: This repository focuses on traditional computer vision techniques and demonstrates the power of classical image processing methods without relying on machine learning or deep learning approaches.