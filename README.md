# MRID - Medical Imaging DICOM Classification Tool

A comprehensive GUI application for automated classification and organization of MRI DICOM series using machine learning models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Conda Environment Setup](#conda-environment-setup)
  - [External Dependencies](#external-dependencies)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Classification Modes](#classification-modes)
  - [Model Types](#model-types)
  - [Training Models](#training-models)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

MRID is a powerful tool designed to automate the classification and organization of brain MRI DICOM series. It uses machine learning models (Random Forest for DICOM headers and CNN for image data) to identify and label common MRI sequences such as T1, T2, FLAIR, and T1 post-contrast.

**Key Capabilities:**
- Automated DICOM series classification
- Intelligent 3D acquisition preference
- DICOM header and image-based classification models
- Interactive GUI for manual and semi-automated classification
- Automatic NIfTI conversion
- Configurable target series and preferences

---

## Features

### Core Functionality
- ✅ **Multi-Modal Classification**: DICOM header model, image-based CNN, or combined
- ✅ **Three Classification Modes**: Manual, Semi-Auto, Full-Auto
- ✅ **Smart Series Selection**: Automatic selection with configurable confidence thresholds
- ✅ **3D Preference**: Prioritize 3D acquisitions over 2D when configured
- ✅ **NIfTI Conversion**: Integrated dcm2niix conversion with automatic organization
- ✅ **Hierarchical Organization**: Study/Series hierarchy with metadata extraction

### Machine Learning
- ✅ **Random Forest Header Model**: configurable classifier using DICOM metadata
  - ✅ **Automatic DICOM Tag Encoding**: Intelligent DICOM header tag encoding for optimal classification
  - ✅ **Semantic Text Encoding**: Sentence transformers for series descriptions
- ✅ **CNN Image Model**: Custom architecture for center slice-based classification
  - ✅ **Data Augmentation**: On-the-fly augmentation during CNN training
  - ✅ **Parallel Processing**: Multi-core preprocessing for faster training

### User Interface
- ✅ **Modern GUI**: Tkinter-based with progress tracking
- ✅ **Series Visualization**: Thumbnail previews of center slices
- ✅ **Color-Coded Menus**: Visual differentiation of series types
- ✅ **Batch Operations**: Process multiple studies efficiently
- ✅ **Utility Menu**: NIfTI conversion and data management tools

---

## Installation

### Prerequisites

**System Requirements:**
- Linux (tested on Ubuntu 20.04+)
- Python 3.8 or higher
- 8GB+ RAM (16GB+ recommended for CNN training)
- CUDA-capable GPU (optional, for faster CNN training)

**Required Software:**
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [dcm2niix](https://github.com/rordenlab/dcm2niix) (for NIfTI conversion)

### Conda Environment Setup

#### 1. Create Conda Environment

```bash
# Clone the repository
git clone https://github.com/ecalabr/mrid.git
cd mrid

# Create conda environment from requirements
conda create -n mrid python=3.12 -y
conda activate mrid
```
#### 2. Install pytorch (GPU version recommended)
https://pytorch.org/get-started/locally/

#### 3. Install other dependencies
```bash
pip install -r requirements.txt
```

#### 2. Requirements File

The `requirements.txt` includes:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
pydicom>=2.3.0
Pillow>=9.0.0
pyyaml>=6.0
joblib>=1.1.0
sentence-transformers>=2.2.0
```

### External Dependencies

#### dcm2niix Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install dcm2niix
```

**macOS (Homebrew):**
```bash
brew install dcm2niix
```

**From Source:**
```bash
git clone https://github.com/rordenlab/dcm2niix.git
cd dcm2niix
mkdir build && cd build
cmake ..
make
sudo make install
```

**Verify Installation:**
```bash
dcm2niix -version
```

### Verify Installation

```bash
# Activate environment
conda activate mrid

# Run tests
python run_tests.py

# Expected output:
# ✓ All tests passed!
# Passed: 32, Failed: 0, Errors: 0
```

---

## Quick Start

### 1. Launch the Application

```bash
conda activate mrid
python run.py
```

### 2. Select Directories

- **Input DIR**: Directory containing DICOM files to classify
- **Output DIR**: Directory where results will be saved (should be empty initially)

### 3. Choose Classification Mode

- **Manual**: Review and classify each series manually
- **Semi-Auto**: Pre-select best series matches based on model predictions, review before saving (requires trained models)
- **Full-Auto**: Automatically classify and save without review (requires trained models)

### 4. Start Classification

1. Click **"▶ Start Classification"**
2. Review classifications and make necessary changes (if not Full-Auto)
3. Click **"Approve"** to save results

### 5. Optional: Convert to NIfTI

- Enable **"Convert to NIfTI: Yes"** before starting
- Or use **"Other Options → Convert NIfTIs"** after classification

---

## Usage

### Classification Modes

#### Manual Mode
- **Use Case**: Maximum control, review every series
- **Workflow**:
  1. System scans DICOM hierarchy
  2. Displays all series with thumbnails
  3. User selects classification from dropdown for each
  4. Click "Approve" to save

#### Semi-Auto Mode
- **Use Case**: Efficient classification with human oversight
- **Workflow**:
  1. System predicts classifications using models
  2. Pre-selects best match per target series
  3. User reviews and adjusts selections
  4. Click "Approve" to save
- **Features**:
  - Automatic duplicate detection
  - Probability display
  - Color-coded selections
  - Override threshold filtering

#### Full-Auto Mode
- **Use Case**: Batch processing, high-volume workflows
- **Workflow**:
  1. System predicts and selects automatically
  2. Saves results without user review
  3. Processes all studies in queue
- **Best For**: Large datasets, consistent imaging protocols

### Model Types

#### DICOM Header Model
- **Description**: Random Forest classifier using DICOM metadata
- **Features Used**:
  - Series descriptions
  - Timing parameters (TR, TE, TI)
  - Sequence information
  - Acquisition properties
  - Spatial information
- **Advantages**:
  - Fast prediction
  - Interpretable features
  - No image loading required
- **Training Time**: <1 minutes for 1000 series

#### Image-Based Model
- **Description**: Convolutional Neural Network using center slices
- **Architecture**: multi-layer CNN with dropout
- **Input**: user tunable resolution grayscale center slice
- **Advantages**:
  - Visual feature learning
  - Robust to metadata inconsistencies
- **Training Time**: ~5-10 minutes for 1000 series (GPU)

#### Header + Image Model
- **Description**: Ensemble combining both models
- **Method**: Average of RF and CNN probabilities
- **Advantages**:
  - Best accuracy
  - Robust to various data quality issues
- **Recommended**: For best performance

### Training Models

#### Prerequisites
- First you need to prepare some training data (using Manual mode at first, +/- Semi-Auto mode)
- Minimum ~50 examples per target class recommended
- More data = better performance

#### Training Workflow

1. **Classify Training Data**
   ```
   - Use Manual or Semi-Auto mode
   - Classify multiple studies
   - Ensure diverse examples per class
   ```

2. **Train Models**
   ```
   - Click "Train Model(s)" button
   - Select model type (Header/Image/Both)
   - Training starts automatically
   - Progress shown in popup window
   ```

3. **Load Models**
   ```
   - Ensure that either Semi-Auto or Full-Auto mode is selected
   - Click "Load Model(s)" button
   - Models loaded for Semi-Auto/Full-Auto use
   ```

4. **Evaluate Performance**
   ```
   - Check training summary files:
     - rf_model_training_summary.txt
     - cnn_model_training_summary.txt
   - Test on validation studies
   ```

#### Training Tips

- **Data Quality**: Accurate labels crucial for good performance
- **Class Balance**: Similar number of examples per class
- **Data Diversity**: Include various scanners, protocols
- **Iterative Improvement**: Retrain with corrected classifications
- **DICOM Tag Additions**: Add new DICOM fields in config.py to improve RF model

---

## Configuration

Configuration is managed in `mrid_app/config.py`.

### Target Series

Define which series types to classify:

```python
TARGET_SERIES = ["T1", "T2", "FLAIR", "T1c", "Other"]
```

Add or remove series as needed. "Other" should always be included.

### 3D Preference

Prefer 3D acquisitions when multiple series exceed the classification threshold:

```python
PREFER_3D = {
    "T1": True,      # Prefer 3D T1
    "T2": False,     # Don't prefer 3D for T2
    "FLAIR": True,   # Prefer 3D FLAIR
    "T1c": True,     # Prefer 3D post-contrast
    "Other": False   # N/A
}
```

When `True`: If multiple series exceed `OVERRIDE_THRESHOLD` for a label, select the 3D acquisition even if it has lower probability.

### Classification Thresholds

```python
OVERRIDE_THRESHOLD = 0.5  # Minimum probability for auto-selection
```

- Higher values: More conservative, fewer auto-selections
- Lower values: More aggressive, may include uncertain matches
- Range: 0.0 to 1.0
- Recommended: 0.4 to 0.6

### DICOM ImageType Filtering

```python
INCLUDE_TYPES = ["ORIGINAL", "PRIMARY"]
EXCLUDE_TYPES = ["DERIVED", "SECONDARY", "ADC", "DIFFUSION", "LOCALIZER", "MIP", "MPR"]
```

Control which ImageTypes are included in classification:
- **INCLUDE_TYPES**: Must have at least one of these
- **EXCLUDE_TYPES**: Excluded if has any of these

### Series Separation

```python
IGNORE_MULTIPLE_ORIENTATIONS = True  # Skip series with multiple orientations
SEPARATE_STACKED_SERIES = True       # Split stacked series
MAX_SPLIT = 5                        # Max splits per stacked series
```

### CNN Hyperparameters

```python
IMAGE_SIZE = 128      # Input image size (128×128)
NUM_EPOCHS = 100      # Training epochs
BATCH_SIZE = 8        # Batch size for training
LR = 0.001            # Learning rate
```

Adjust based on your data and computational resources.

### Random Forest Hyperparameters

```python
N_ESTIMATORS = 400    # Number of trees in forest
```

More trees may yeild better performance but slower training.

### DICOM Header Fields

Specify which DICOM fields to use for header model:

```python
DICOM_HEADER_FIELDS = [
    ("SeriesDescription", "SeriesDescription"),
    ("TR (Repetition Time)", "RepetitionTime"),
    ("TE (Echo Time)", "EchoTime"),
    # ... add more fields
]
```

**Format**: `(display_name, DICOM_attribute_name)` using pydicom attribute names.

**Adding New Fields**:
1. Add to `DICOM_HEADER_FIELDS`
2. Train model → System prompts to update existing CSVs
3. Confirm to extract field from previous studies

**Automatic Type Detection**:
- System infers field type (numerical, array, categorical, text)
- Applies appropriate encoding automatically
- See `field_detection.py` for logic

### Field Detection Thresholds

```python
CATEGORICAL_UNIQUE_THRESHOLD = 50  # ≤50 unique values = categorical
TEXT_MIN_WORDS = 2                  # ≥2 avg words = text
ARRAY_SIZE_THRESHOLD = 1            # >1 number = array
```

Fine-tune automatic field type detection.

---

## Project Structure

```
mrid/
├── mrid_app/               # Main application package
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── gui.py              # Tkinter GUI and main logic
│   ├── processing.py       # DICOM processing and classification
│   ├── scanner.py          # DICOM hierarchy scanning
│   ├── header.py           # Random Forest header model
│   ├── cnn.py              # CNN image model
│   ├── prep.py             # Data preprocessing
│   ├── field_detection.py  # Automatic field type detection
│   ├── csv_field_updater.py# CSV field management
│   └── utils.py            # Utility functions
├── tests/                  # Unit tests
│   ├── test_config.py
│   ├── test_field_detection.py
│   ├── test_processing.py
│   ├── test_csv_field_updater.py
│   └── test_prep.py
├── run.py                  # Application entry point
├── run_tests.py            # Test runner
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Advanced Features

### File Options Menu

Access via menu bar: **"File Options"**

#### Convert NIfTIs
- **Purpose**: Batch convert all classified series to NIfTI format
- **Usage**: Select menu item → Confirm → Wait for completion
- **Features**:
  - Skips already converted series
  - Shows progress bar
  - Reports success/skip/error counts
- **Output**: NIfTI files saved in study directories with naming `{study_id}_{label}.nii.gz`

#### Erase Data
- **Purpose**: Clean up output directory
- **Options**:
  - ☐ Delete Classifications (study folders, NIfTIs, CSVs)
  - ☐ Delete Models (trained models, encoders, cache)
  - ☐ Delete Hierarchy (hierarchy.yaml, temporary files)
- **Safety**: Multiple confirmation dialogs
- **Use Cases**: Fresh start, disk space management, retraining

### CSV Field Management

When adding new DICOM header fields:

1. **Detection**: System scans existing CSVs for missing fields
2. **Prompt**: Dialog asks to update CSVs with new fields
3. **Extraction**: Reads original DICOM files to get values
4. **Update**: Updates all classification CSVs
5. **Training**: Proceeds with complete dataset

This ensures models always use all configured fields.

### Parallel Processing

CNN preprocessing uses multiprocessing for speed:
- **Workers**: CPU count - 1 (configurable)
- **Speedup**: ~10-30x depending on system
- **Progress**: Real-time updates during processing

### Model Caching

Preprocessed training data cached for fast retraining:
- **Location**: `{output_dir}/cnn_preprocessed_data_{hash}.pkl`
- **Content**: Preprocessed tensors ready for training
- **Benefits**: Skip preprocessing on subsequent training runs
- **Invalidation**: Automatic if source data changes

---

## Troubleshooting

### Common Issues

#### "Missing required dependencies"
**Solution**: Ensure all packages installed:
```bash
pip install -r requirements.txt
```

#### "dcm2niix not found"
**Solution**: Install dcm2niix and add to PATH:
```bash
sudo apt-get install dcm2niix
# or
brew install dcm2niix
```

#### "CUDA out of memory" during CNN training
**Solution**: Reduce batch size in config:
```python
BATCH_SIZE = 4  # or 2
```

#### "No classification CSV files found"
**Solution**: Classify some studies first using Manual or Semi-Auto mode before training.

#### Models not loading
**Solution**: Check files exist in output directory:
- `rf_classifier_model.joblib`
- `label_encoder.joblib`
- `feature_encoder.joblib`
- `cnn_classifier_model.pth` (for CNN)

#### Slow CNN training
**Solutions**:
- Install PyTorch with CUDA support
- Reduce `IMAGE_SIZE` (e.g., 64 instead of 128)
- Reduce `NUM_EPOCHS`
- Use fewer training examples initially

#### GUI freezes
**Solution**: Processing happens in background threads, but very large studies may cause brief freezes. Be patient or process in smaller batches.

### Getting Help

1. **Check TESTING.md**: For development and testing guidance
2. **Run Tests**: `python run_tests.py` to verify installation
3. **Check Logs**: Terminal output provides detailed error messages
4. **Open Issue**: Report bugs on GitHub with:
   - System info (OS, Python version)
   - Error messages
   - Steps to reproduce

---

## Testing

Run the test suite:

```bash
python run_tests.py
```

Expected output:
```
✓ All tests passed!
Passed: 32, Failed: 0, Errors: 0
Time: ~1.85s
```
---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run test suite (`python run_tests.py`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{mrid2026,
  title={MRID: Medical Imaging DICOM Classification Tool},
  author={Evan Calabrese},
  year={2026},
  url={https://github.com/ecalabr/mrid}
}
```
Please check back for updates to citation information.

---

## Acknowledgments

- **pydicom**: DICOM file handling
- **dcm2niix**: NIfTI conversion
- **sentence-transformers**: Semantic text encoding
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities


---

**Last Updated**: January 2026  
**Version**: 1.0.0
