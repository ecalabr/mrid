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
    - [Analyze Mode](#analyze-mode)
    - [Manual Mode](#manual-mode)
    - [Semi-Auto Mode](#semi-auto-mode)
    - [Full-Auto Mode](#full-auto-mode)
    - [Verify Mode](#verify-mode)
  - [Model Types](#model-types)
  - [Training Models](#training-models)
- [Basic Configuration](#basic-configuration)
- [Detailed Configuration Options](#detailed-configuration-options)
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
- ✅ **Five Classification Modes**: Analyze, Manual, Semi-Auto, Full-Auto, Verify
- ✅ **Analyze Mode**: Fully automated series discovery using deep learning — no pre-defined labels required; can also pre-train DICOM header and CNN models from the discovered clusters with **zero manual labeling**
- ✅ **Smart Series Selection**: Automatic selection with configurable confidence thresholds
- ✅ **3D Preference**: Prioritize 3D acquisitions over 2D when configured
- ✅ **NIfTI Conversion**: Integrated dcm2niix conversion with automatic organization
- ✅ **Hierarchical Organization**: Study/Series hierarchy with metadata extraction
- ✅ **Compressed DICOM Support**: Handles JPEG, JPEG2000, RLE compressed files via python-gdcm
- ✅ **Automatic Folder Organization**: Detects and organizes unstructured DICOM folders using symlinks
- ✅ **Organized Output Structure**: Classified series outputs (study folders, CSVs, NIfTIs) are stored in a dedicated `classified_outputs/` subfolder inside the output directory, keeping scan metadata files separate

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
python-gdcm  # For compressed DICOM support
Pillow>=9.0.0
pyyaml>=6.0
joblib>=1.1.0
sentence-transformers>=2.2.0
umap-learn   # For Analyze mode dimensionality reduction
hdbscan      # For Analyze mode clustering
```

**Note on Compressed DICOMs:** The application uses `python-gdcm` to handle compressed DICOM files (JPEG, JPEG2000, RLE, etc.). This library provides native support for reading compressed transfer syntaxes that pydicom alone cannot handle.

**Note on Analyze Mode:** `umap-learn` and `hdbscan` are required for Analyze mode. They are not needed for Manual, Semi-Auto, Full-Auto, or Verify modes.

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
python tests.py

# Expected output:
# ✓ All tests passed!
# Passed: 48, Failed: 0, Errors: 0
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

- **Analyze**: Automatically discover series types from your dataset
  - This mode is optional. You can skip it if you already know your `TARGET_SERIES` and have them configured in `config.py`
  - Use this mode at the very beginning if you need help determining your `TARGET_SERIES` list in `config.py`
  - Optionally, use **"Train Model(s)"** after analysis to pre-train DICOM header and CNN models from the discovered clusters — no manual labeling required
- **Manual**: Review and classify each series manually
- **Semi-Auto**: Pre-select best series matches based on model predictions, review before saving (requires trained models)
- **Full-Auto**: Automatically classify and save without review (requires trained models)
- **Verify**: Review and validate already-classified studies, mark incorrect classifications for deletion

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

#### Analyze Mode
- **Use Case**: First-time dataset exploration, automatic `TARGET_SERIES` configuration, and — uniquely — **pre-training DICOM header and CNN models from scratch with zero manual labeling**. Use this mode when you have a new dataset and want to go from raw DICOMs to working classification models without labeling a single study by hand.
- **Workflow**:
  1. Select Input and Output directories and set Mode to **Analyze**
  2. Click **"▶ Start Analysis"**
  3. System scans DICOM files and generates series thumbnails (skipped if already done)
  4. Image series are automatically clustered into similar groups. Use the `ANALYZE_COVERAGE_THRESHOLD` parameter in `config.py` to define the minimum fraction of exams that must contain an example for a given cluster to be considered valid
  5. A results window opens showing the 4 most-representative thumbnails per cluster, with a suggested name pre-populated from the series descriptions of high-confidence cluster members
  6. For each cluster, review or edit the series name and optionally tick **"Prefer 3D"** if you want the system to automatically prefer 3D volumetric images for this series type
  7. Uncheck **"Include"** for any series cluster you do not want to include in your workflow
  8. Click **"Apply"** — `TARGET_SERIES` and `PREFER_3D` are written to `config.py` in the output directory, and a `selfsupervised_classification_data.csv` file is written to `{output_dir}/model_files/` containing high-confidence cluster members for all approved clusters
  9. **(Optional) Train models without any manual labeling**: Once analysis is complete, the **"Train Model(s)"** button becomes active. Click it, select the desired model type (Header / Image / Both), and training will proceed automatically using only the high-confidence series from `selfsupervised_classification_data.csv`. No additional labeled data is needed.
  10. After training, switch to **Semi-Auto** or **Full-Auto** mode, load the newly trained models, and begin classifying your dataset
- **Zero-Label Training Pipeline** (the full end-to-end flow):
  ```
  Analyze mode → cluster discovery → name clusters → Apply
       ↓
  selfsupervised_classification_data.csv written automatically
       ↓
  Train Model(s) (Header / CNN / Both) — uses CSV above, no manual labels
       ↓
  Load Model(s) → switch to Semi-Auto or Full-Auto → classify dataset
  ```
- **Technical Pipeline**:
  - **Feature extraction**: Frozen ResNet50 (ImageNet weights, already cached), 2048-D global average-pool output, batched GPU inference (~2 ms/image on modern hardware)
  - **L2 normalization**: Makes cosine distance equivalent to Euclidean distance on the unit hypersphere
  - **PCA**: Reduces 2048-D features to the minimum number of components that retain ≥ 90 % of variance (typically 3–15 for MRI datasets), capped at 150
  - **UMAP**: Non-linear manifold embedding preserving global topology (`n_neighbors=30`, `min_dist=0.1`, cosine metric); output dimension = min(PCA dims, 15)
  - **HDBSCAN**: Density-based clustering with `min_cluster_size = ceil(ANALYZE_COVERAGE_THRESHOLD × n_exams)`; noise points are reassigned to the nearest cluster centroid
  - **Cluster naming**: High-confidence members (HDBSCAN soft probability > 0.8) are identified; `SeriesDescription` is extracted from the corresponding DICOM files; a medoid using pairwise Levenshtein distance selects the most representative description as the suggested name
  - **Best examples**: The 4 thumbnails per cluster closest to the centroid in PCA space are shown in the results window
  - **Training data collection**: After the user approves clusters, high-confidence members from approved clusters are collected into `selfsupervised_classification_data.csv` (same format as standard `classification_data.csv`)
- **Output**:
  - `TARGET_SERIES` and `PREFER_3D` updated in `{output_dir}/config.py`
  - `"Other"` is always appended as the final catch-all entry regardless of user selections
  - `{output_dir}/model_files/selfsupervised_classification_data.csv` — high-confidence labeled training data, ready for immediate model training
- **Notes**:
  - Requires `torch`, `torchvision`, `umap-learn`, and `hdbscan` packages
  - ResNet50 weights are loaded from the local PyTorch cache — no internet access required after first run
  - Scanning and thumbnail generation are skipped automatically if already completed
  - If no `config.py` is found in the output directory, you will be prompted to copy the default
  - The **"Train Model(s)"** button is grayed out (with a tooltip *"Analyze first"*) until `selfsupervised_classification_data.csv` has been written; it activates automatically once analysis is complete

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

#### Verify Mode
- **Use Case**: Quality control and validation of classified studies
- **Workflow**:
  1. System scans output directory for all classified studies
  2. Aggregates classifications by target series
  3. Displays batches of thumbnails grouped by target series classification
  4. User reviews and marks incorrect classifications
  5. Option to delete entire output folders for incorrect classifications
  6. Allows subsequent re-classification of incorrect studies
- **Features**:
  - Batch display (configurable grid size)
  - Toggle buttons: "Correct" (default) → "Incorrect" (red)
  - Confirmation before deletion
- **Best For**: Post-processing validation, training data cleanup, quality assurance
- **Note**: Does not require models - works with any classified data

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
- You need labeled training data to train models. There are **two ways** to obtain this:
  1. **Manual labeling** (traditional): Classify studies using Manual or Semi-Auto mode first
  2. **Zero-label (Analyze mode)**: Run Analyze mode — the app automatically collects high-confidence cluster members as training data with no manual labeling required
- Minimum ~50 examples per target class recommended
- More data = better performance

#### Zero-Label Training Workflow (Analyze Mode — no manual labeling required)

This is the recommended starting point for new datasets where no labeled data exists:

1. **Run Analyze Mode**
   ```
   - Set mode to Analyze, click "▶ Start Analysis"
   - Name the discovered clusters in the results window
   - Click "Apply" to write config.py and collect training data
   - selfsupervised_classification_data.csv is written automatically to
     {output_dir}/model_files/
   ```

2. **Train Models Directly from Analyze Results**
   ```
   - The "Train Model(s)" button activates once analysis is complete
   - Click "Train Model(s)", select model type (Header / Image / Both)
   - Training uses selfsupervised_classification_data.csv (high-confidence
     cluster members only) — no manual labels required
   - Progress is shown in a popup window
   ```

3. **Load Models and Classify**
   ```
   - Switch to Semi-Auto or Full-Auto mode
   - Click "Load Model(s)" to load the newly trained models
   - Begin classifying your full dataset
   ```

4. **Iterate and Improve** *(optional)*
   ```
   - Use Semi-Auto mode to annotate data with model assistance
   - Re-train models with the growing body of verified labels
   - Performance improves with each iteration
   ```

#### Standard Training Workflow (Manual Labeling)

#### Prerequisites (Standard)
- First you need to prepare some training data (using Manual mode at first, +/- Semi-Auto mode)
- Minimum ~50 examples per target class recommended
- More data = better performance


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

## Basic Configuration

Configuration is managed in `mrid_app/config.py`.

### Configuration Recommendations
You should adjust the following settings for each project:
- TARGET_SERIES: Define series types relevant to your workflow
- PREFER_3D: Set preferences for 3D acquisitions
- STUDY_NAME_TEMPLATE: Customize output naming convention

Everything else can be left at default values unless you have specific needs.

## Detailed Configuration Options

### Default Directories
The app automatically remembers the last-used input and output directories between sessions (stored in `.mrid_prefs.json` in the project root directory, alongside `run.py`). On first run it defaults to the current user's home directory.

### Output Naming Convention
Outputs are named using the following template:
```python
STUDY_NAME_TEMPLATE = "{mrn}-{study_uid}"
```
Customize the output naming convention using any combination of MRN, Accession Number, and Study Instance UID as follows:
- {mrn} = Medical Record Number
- {study_uid} = Study Instance UID
- {accn} = Accession Number

Examples:
```python
STUDY_NAME_TEMPLATE = "MyProject-{mrn}-{study_uid}-{accn}"
STUDY_NAME_TEMPLATE = "{mrn}_{accn}"
STUDY_NAME_TEMPLATE = "{study_uid}"
```
Notes:
- Ensure the template results in unique names for each study to avoid overwriting.
- Use of MRN as the only identifier may lead to conflicts if multiple studies exist for the same patient.
- Invalid characters will be replaced with hyphens.

### Thumbnail options
```python
THUMBNAIL_FORMAT = 'JPEG'  # Format for generated thumbnails: 'JPEG' or 'PNG'
THUMBNAIL_QUALITY = 75  # JPEG quality (1-100, only used if THUMBNAIL_FORMAT='JPEG')
```
- Controls thumbnail format for GUI display only (CNN model uses separate preprocessing).
- Thumbnail format can be either 'JPEG' or 'PNG'.
- Thumbnail quality applies only to JPEG format, ranges from 1-100, and controls compression level (higher = better quality, larger file size).
- Recommended settings are JPEG format with quality of 75 for a good balance between quality and file size.

### Target Series

Define which series types to classify:

```python
TARGET_SERIES = ["T1", "T2", "FLAIR", "T1c", "Other"] # always include "Other"
```

- You can use whatever series names you want as long as there are no special characters.
- These are just labels and don't need to match the series description.
- Add or remove series as needed.
- "Other" should always be included (it's the default for unclassified series).
- In **Analyze mode**, this list is populated automatically and written to `config.py` — you do not need to set it manually.

### Analyze Mode Coverage Threshold

Controls how inclusive or exclusive the automatic series-type discovery in Analyze mode is:

```python
ANALYZE_COVERAGE_THRESHOLD = 0.1  # default: 10% of exams must contain a series type
```

- **Meaning**: A series type is only recognised as a valid cluster if it appears in at least this fraction of exams in the dataset.
- **How it works**: The value is passed directly to HDBSCAN as `min_cluster_size = ceil(ANALYZE_COVERAGE_THRESHOLD × n_unique_exams)`. Any group of thumbnails smaller than this threshold is treated as noise and merged into the nearest valid cluster.
- **Effect of increasing the value** (e.g. `0.5`): Only series types present in ≥ 50 % of exams are reported — rare or incidental sequences are suppressed. Use when your protocol is highly consistent and you only want the core series types.
- **Effect of decreasing the value** (e.g. `0.05`): Series types present in as few as 5 % of exams are reported — more clusters, including rare or optional sequences. Use when your dataset contains heterogeneous protocols or you want to capture all series types.
- **Range**: 0.0–1.0 (values below ~0.05 may produce many small clusters; values above ~0.5 may merge similar types)
- **Recommended starting point**: `0.1` (10 %) for most datasets, higher if you are getting too many series options in analyze mode, lower if you want more options

### 3D Preference

Prefer 3D acquisitions when multiple series exceed the classification threshold.

`PREFER_3D` can be set in two ways:

**Per-series dict** (recommended) — explicit `True`/`False` for each series you care about:
```python
PREFER_3D = {
    "T1": True,      # Prefer 3D T1
    "T2": False,     # Don't prefer 3D for T2
    "FLAIR": True,   # Prefer 3D FLAIR
    "T1c": True,     # Prefer 3D post-contrast
    "Other": False   # N/A (always ignored for Other)
}
```
- Any series **missing** from the dict defaults to `False` (no 3D preference).
- Any **extra** keys in the dict (series not in `TARGET_SERIES`) are silently ignored.

**Global bool** — applies the same preference to every series except `Other`:
```python
PREFER_3D = True   # Prefer 3D for all non-Other series
# or
PREFER_3D = False  # Never prefer 3D (default behaviour)
```

When the preference for a label is `True`: if multiple series exceed `OVERRIDE_THRESHOLD` for that label, the 3D acquisition is selected even if it has a slightly lower probability than a 2D candidate.

### Classification Thresholds

```python
OVERRIDE_THRESHOLD = 0.5  # Minimum probability for auto-selection
```

- Higher values: More conservative, fewer auto-selections
- Lower values: More aggressive, may include uncertain matches
- Range: 0.0 to 1.0
- Recommended: 0.4 to 0.6

### Label Uniqueness

```python
ALLOW_DUPLICATE_LABELS = True
```

Controls whether manually selecting a classification label for one series automatically clears that label from other series in the same study:

- **`False` (Default)**: Enforces one series per label per study
  - When you select a label (e.g., "T1") for a series, any other series with that label is automatically reset to "Other"
  - Useful for standard classification workflows where each study should have only one series per type
  - Applies to both Manual and Semi-Auto modes

- **`True`**: Allows multiple series per label
  - You can manually assign the same label to multiple series in a study
  - Note that auto-selection in Semi-Auto mode will still only select one best match per label based on model probabilities
  - Useful for creating training data with multiple examples per label from the same study
  - Both Header and Image models support training on data with duplicate labels

### UI Grid Layout

```python
NUM_COLUMNS = 6          # Number of columns in classification/verification grid
NUM_ROWS = 4             # Max number of rows in verification mode grid
THUMBNAIL_SCALE_FACTOR = 1.0  # Additional static scale multiplier for thumbnail size only
```

Controls the grid layout for classification and verification interfaces:

- **NUM_COLUMNS**: Maximum number of series displayed horizontally (applies to all modes)
- **NUM_ROWS**: Maximum number of series displayed vertically in Verify mode
- **Grid Size**: In Verify mode, up to `NUM_COLUMNS × NUM_ROWS` images displayed per batch
- **THUMBNAIL_SCALE_FACTOR**: Multiplies only the thumbnail (image panel) pixel size on top of the automatic DPI scaling. Does not affect fonts, buttons, or other non-thumbnail elements.
  - `1.0` (default) — no additional scaling
  - `0.75` — 25 % smaller thumbnails, more thumbnails visible without scrolling
  - `1.5` — 50 % larger thumbnails for high-resolution monitors
- **Vertical scrollbar**: Both the classification (Manual/Semi-Auto) and Verify windows always include a vertical scrollbar so any content that extends beyond the screen height remains accessible. The windows are also capped to the screen height to ensure the bottom button row is never hidden off-screen.
- **Recommendations**:
  - Standard monitors: 6 columns × 3-4 rows (18-24 images)
  - Large monitors: 8 columns × 5-6 rows (40-48 images)
  - Adjust based on screen resolution and preference

### Parallel Image Loading

```python
NUM_PARALLEL_WORKERS = 8  # Number of parallel threads for loading thumbnails
```

Controls the number of concurrent worker threads used to load image thumbnails in Semi-Auto and Verify modes:

- **Purpose**: Optimizes thumbnail loading performance through parallel processing
- **Default**: 8 workers (suitable for most modern systems, adjust as needed based on your system capabilities)

### Study Order Randomization

```python
RANDOMIZE_STUDY_ORDER = True  # Randomize the order of studies during manual/semi-auto classification
STUDY_ORDER_SEED = 42         # Seed for pseudo-random study ordering (ensures reproducibility)
```

Controls whether studies are presented in a randomized order during Manual and Semi-Auto classification:

- **Purpose**: Reduces bias in manual classification by randomizing the order studies are presented
- **`RANDOMIZE_STUDY_ORDER`**: Enable/disable randomization
  - `True` (Default): Studies are shuffled in pseudo-random order
  - `False`: Studies are presented in the order found (patient ID, then study UID)
- **`STUDY_ORDER_SEED`**: Integer seed for random number generator
  - Default: 42
  - Using the same seed ensures reproducible ordering across runs
  - Change the seed to get a different random order
  - Only used when `RANDOMIZE_STUDY_ORDER = True`
- **Note**: Only applies to Manual and Semi-Auto modes; Full-Auto mode processes studies in original order for consistency

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
│   ├── analyze.py          # Analyze mode: ResNet50 → PCA → UMAP → HDBSCAN pipeline
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

**Output directory layout** (created at runtime inside your chosen Output DIR):
```
{output_dir}/
├── config.py                        # Project-specific config (copied from repo on first run)
├── hierarchy.yaml                   # DICOM hierarchy index
├── series_thumbnails/               # Center-slice thumbnails used by the GUI and Analyze mode
├── dicom_symlinks/                  # Symlinks for unorganized DICOM folders (if needed)
├── model_files/                     # Trained models and Analyze mode training data
│   ├── rf_classifier_model.joblib
│   ├── cnn_classifier_model.pth
│   ├── label_encoder.joblib
│   ├── feature_encoder.joblib
│   ├── classification_data.csv      # Aggregated classification results for model training
│   └── selfsupervised_classification_data.csv      # Written by Analyze mode
└── classified_outputs/              # All classification results live here
    └── {study_id}/                  # One folder per classified study
        ├── {study_id}_classification-results.csv   # Classification results for this study
        └── {study_id}_{series_name}.nii.gz         # NIfTI file (if conversion enabled)
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

### Configuration Menu

Access via menu bar: **"Configuration"**

#### Save Config File
- **Purpose**: Save current configuration to output directory
- **Usage**: Select menu item → Config saved to `{output_dir}/config.py`
- **Use Cases**:
  - Preserve project-specific settings
  - Share configurations across datasets
  - Document configuration used for specific analyses
- **Validation**: Checks that output directory exists before saving

#### Load Config File
- **Purpose**: Load a previously saved configuration file
- **Usage**: Select menu item → Choose config.py file → Config loaded
- **Features**:
  - File browser for config selection
  - Automatic validation of all required settings
  - Warning if invalid (continues with original config)
  - Updates all configuration globals dynamically
- **Validation**: Ensures loaded config contains all required variables from repository config
- **Automatic Loading**: When starting classification, system checks output directory for saved config.py:
  - If found alongside hierarchy.yaml: Prompts to load both
  - If config invalid: Shows warning, uses original config
  - Seamless integration with existing hierarchy workflow
- **Note**: Some settings may require application restart to fully take effect

#### Edit Config File
- **Purpose**: Open the currently active config.py file in your system's default text editor
- **Features**:
  - Opens repository config.py or custom loaded config
  - Uses system default text editor (Notepad, TextEdit, gedit, etc.)
  - Shows which config is active (repository vs. custom)
- **Usage**: Select menu item → Config file opens in editor → Make changes → Save file → Use "Reload Config"
- **Note**: After editing, use "Configuration > Reload Config" to apply changes

#### Reload Config File
- **Purpose**: Reload configuration from the currently active config file
- **Features**:
  - Reloads either repository config or custom loaded config
  - Updates all configuration variables in memory
  - Validates config before applying
- **Usage**: After editing config.py, select menu item to apply changes
- **Note**: Some settings (loaded models, running processes, certain UI elements) may require application restart to fully take effect


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

### Compressed DICOM Support

The application automatically handles compressed DICOM files using **python-gdcm**:
- **Supported Formats**: JPEG, JPEG2000, JPEG-LS, RLE, and other compressed transfer syntaxes
- **Fallback Mechanism**: If pydicom cannot read a file, gdcm is automatically attempted
- **Transparent Operation**: No user intervention needed - compressed files are processed seamlessly
- **Performance**: Efficient decompression without creating temporary uncompressed files

**Note**: python-gdcm is automatically installed with the application dependencies.

### Automatic Folder Organization

The application can automatically detect and organize unstructured DICOM folders:

**Organized Folders** (no action needed):
```
input/
  ├── Patient1/
  │   ├── Study1/
  │   │   ├── Series1/  # Each series in separate folder
  │   │   │   ├── image1.dcm
  │   │   │   └── image2.dcm
  │   │   └── Series2/
  │   │       ├── image1.dcm
  │   │       └── image2.dcm
```

**Unorganized Folders** (automatically organized):
```
input/
  ├── all_files_mixed/  # Multiple series in same folder
  │   ├── series1_image1.dcm
  │   ├── series1_image2.dcm
  │   ├── series2_image1.dcm
  │   └── series2_image2.dcm
```

When unorganized folders are detected:
1. System creates `dicom_symlinks/` directory in output folder
2. Files are organized into series-specific subdirectories using **symlinks**
3. Symlinks point to original files (no duplication)
4. Processing continues with organized structure
5. Message displayed: `"Detected unorganized DICOM folder. Organizing files with symlinks..."`

**Benefits**:
- Works with any folder structure (PACS exports, research datasets, etc.)
- No disk space overhead (symlinks only)
- Original files remain untouched
- dcm2niix and other tools work properly with organized structure

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

#### Compressed DICOMs not reading
**Problem**: Files skipped or "Invalid DICOM" errors
**Solution**: 
1. Ensure python-gdcm is installed:
   ```bash
   pip install python-gdcm
   ```
2. Verify file is actually DICOM (check for DICM preamble)
3. Check console for specific error messages
4. Try opening file with pydicom directly to diagnose:
   ```python
   import pydicom
   ds = pydicom.dcmread('path/to/file.dcm', force=True)
   ```

#### "Multiple series in same folder" warning
**Not an Error**: This is informational - the application is organizing your files
- System automatically creates symlinks in `dicom_symlinks/`
- Original files are not modified
- Processing continues normally
- To avoid this, organize DICOMs with one series per folder before input

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
python tests.py
```

Expected output:
```
✓ All tests passed!
Passed: 48, Failed: 0, Errors: 0
Time: ~2.2s
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

**Last Updated**: February 2026  
**Version**: 1.1.0
