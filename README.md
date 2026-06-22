# MRID — Medical Imaging DICOM Classification Tool

A GUI application for automated classification and organization of MRI DICOM series using machine learning models.

[![Version 2.0](https://img.shields.io/badge/version-2.0-brightgreen.svg)](https://github.com/ecalabr/mrid)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Linux](#linux)
  - [macOS](#macos)
  - [Windows](#windows)
  - [PyTorch GPU Setup](#pytorch-gpu-setup)
  - [dcm2niix](#dcm2niix)
  - [Verify Installation](#verify-installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Classification Modes](#classification-modes)
  - [Model Types](#model-types)
  - [Training Models](#training-models)
  - [Series Thumbnail Scrolling](#series-thumbnail-scrolling)
  - [Headless Batch Classification](#headless-batch-classification)
- [Configuration](#configuration)
  - [Config Wizard](#config-wizard)
  - [Configuration Menu](#configuration-menu)
  - [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

MRID automates the classification and organization of brain MRI DICOM series. It uses machine learning models (a Random Forest on DICOM headers and a CNN on image data) to identify and label MRI sequences such as T1, T2, FLAIR, and T1 post-contrast.

**Key capabilities:**
- Manual to fully-automated DICOM series classification with five workflow modes
- Zero-annotation model bootstrapping via unsupervised series discovery (Analyze mode)
- DICOM header and image-based classification models, individually or combined
- Interactive dark-mode GUI for manual and semi-automated classification
- Automatic NIfTI conversion via dcm2niix
- Per-project configuration with a guided setup wizard

---

## Features

### Core Functionality
- **Multi-modal classification**: DICOM header model, image-based CNN, or combined
- **Five classification modes**: Analyze, Manual, Semi-Auto, Full-Auto, Verify
    - **Analyze mode**: Fully automated series discovery — can pre-train both models from unsupervised series clustering with zero manual labeling
    - **Manual mode**: Facilitates fully manual series labeling
    - **Semi-Auto mode**: Trained models pre-select series labels, user reviews and corrects as needed
    - **Full-Auto mode**: Trained models classify automatically without user review
    - **Verify mode**: Rapidly verify and/or correct existing classifications in batches
- **Smart series selection**: Automatic selection with configurable confidence thresholds and optional 3D-acquisition preference
- **NIfTI conversion**: Integrated dcm2niix conversion with automatic file organization
- **Compressed DICOM support**: JPEG, JPEG2000, JPEG-LS, and RLE transfer syntaxes via python-gdcm
- **Automatic folder organization**: Unstructured DICOM folders are organized with symlinks (no file duplication)
- **Broken-filepath repair**: relink a project after its input/output directories are moved or renamed, with the option to locate or drop missing image data
- **User-friendly configuration**: Per-project `config.py` with a guided Config Wizard for non-coders

### Machine Learning
- **Random Forest header model** with automatic DICOM header field type detection and smart encoding such as semantic text encoding (sentence transformers) for free text fields like series descriptions
- **CNN image model** with on-the-fly augmentation, parallel preprocessing, and preprocessing cache
    - **2D or 2.5D CNN input**: classify from the center slice alone or from several evenly spaced slices (see configurable parameter: `CNN_NUM_SLICES`)

### User Interface
- **Modern dark-mode GUI** (CustomTkinter) with a uniform appearance on Windows, macOS, and Linux
- **Screen-proportional scaling**: the UI occupies the same fraction of the screen on any monitor size or resolution
- **Adjustable UI scale**: **View → UI Scale** fine-tunes the overall size of text and controls, applied live and remembered between sessions
- **Consistent window placement**: child windows open in the same cascaded layout every time
- **Scrollable series thumbnails**: mouse-wheel over any thumbnail to page through the full image series
- **Config wizard**: guided, multi-page setup of project configuration files

---

## Installation

MRID runs on Linux, macOS, and Windows. All platforms follow the same outline:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) (recommended but not required)
2. Create the virtual environment and install PyTorch (see [PyTorch GPU Setup](#pytorch-gpu-setup))
3. Install the remaining dependencies with pip
4. Install [dcm2niix](#dcm2niix) (optional — only needed for NIfTI conversion)

**System requirements:** Python 3.8+, 8 GB+ RAM (16 GB+ recommended for CNN training). A PyTorch compatible GPU is optional but speeds up CNN training and Analyze mode considerably.

### Linux

```bash
git clone https://github.com/ecalabr/mrid.git
cd mrid

conda create -n mrid python=3.12 -y
conda activate mrid

# Install PyTorch first — pick the right command for your GPU
# (see "PyTorch GPU Setup" below). CPU-only example:
pip install torch torchvision

pip install -r requirements.txt
```

### macOS

```bash
git clone https://github.com/ecalabr/mrid.git
cd mrid

conda create -n mrid python=3.12 -y
conda activate mrid

# The default PyTorch build includes Apple Silicon (MPS) GPU support
pip install torch torchvision

pip install -r requirements.txt
```

### Windows

Install [Git for Windows](https://git-scm.com/download/win) (or download the repository as a ZIP), then in an **Anaconda Prompt**:

```bat
git clone https://github.com/ecalabr/mrid.git
cd mrid

conda create -n mrid python=3.12 -y
conda activate mrid

REM Install PyTorch first — pick the right command for your GPU
REM (see "PyTorch GPU Setup" below). CPU-only example:
pip install torch torchvision

pip install -r requirements.txt
```

Notes for Windows:
- If `python-gdcm` fails to build from pip, install it from conda-forge instead: `conda install -c conda-forge gdcm`
- Automatic folder organization uses symlinks, which on Windows require either Developer Mode (Settings → Privacy & security → For developers) or running as administrator

### PyTorch GPU Setup

PyTorch is required for the CNN image model and Analyze mode. Install it **before** `pip install -r requirements.txt`, choosing the build that matches your GPU. The commands below are current as of this writing — always check the [official PyTorch install selector](https://pytorch.org/get-started/locally/) for the latest versions.

| Hardware | Platforms | Install command                                                                                                                                                   |
|----------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NVIDIA GPU (CUDA) | Linux, Windows | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` (URL may differ based on your CUDA version, check PyTorch website for details) |
| AMD GPU (ROCm) | Linux only | `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2`                                                                              |
| Apple Silicon (MPS) | macOS | `pip install torch torchvision` (MPS support is built into the default build)                                                                                     |
| CPU only | All | `pip install torch torchvision`                                                                                                                                   |

**Verify GPU support:**
```bash
# NVIDIA / AMD (both report through the CUDA API)
python -c "import torch; print(torch.cuda.is_available())"

# Apple Silicon
python -c "import torch; print(torch.backends.mps.is_available())"
```

Notes:
- **NVIDIA**: install a recent NVIDIA driver first; the pip wheels bundle the CUDA runtime, so a separate CUDA Toolkit install is not required.
- **AMD**: ROCm wheels are Linux-only and report availability through `torch.cuda.is_available()`. Check the [ROCm compatibility list](https://rocm.docs.amd.com/) for supported GPUs.
- **Apple Silicon**: MRID selects the best available device automatically — CUDA (NVIDIA/AMD ROCm) → Apple MPS → CPU — so CNN training and inference use the MPS GPU when available.

### dcm2niix

Required only for NIfTI conversion; classification works without it.

- **Ubuntu/Debian:** `sudo apt-get install dcm2niix`
- **macOS (Homebrew):** `brew install dcm2niix`
- **Windows / any platform (conda):** `conda install -c conda-forge dcm2niix`
- **Windows (manual):** download a [release](https://github.com/rordenlab/dcm2niix/releases) and add the folder containing `dcm2niix.exe` to your `PATH`

Verify with `dcm2niix -version`

### Verify Installation

```bash
conda activate mrid
python tests.py
# Expected: "✓ All tests passed!"
```

---

## Quick Start

1. **Launch the application**
   ```bash
   conda activate mrid
   python run.py
   ```

2. **Select directories**
   - **Input DIR**: directory containing the DICOM files to classify
   - **Output DIR**: directory where results are saved (recommend an empty directory for a new project)

3. **Set up the project config** — when you start a workflow with an output directory with no `config.py`, MRID offers to run the [Config Wizard](#config-wizard) or copy the default config file.

4. **Choose a mode**
   - **Analyze**: discover series types automatically (recommended for new datasets)
   - **Manual**: review and classify each series yourself
   - **Semi-Auto**: model pre-selects, you review (requires trained models)
   - **Full-Auto**: model classifies and saves without review (requires trained models)
   - **Verify**: review already-classified studies

5. **Start** — click **▶ Start Classification**, review (unless Full-Auto), and click **Approve (d)** to save each study.

6. **Optional NIfTI conversion** — set **Convert to NIfTI: Yes** before starting, or run **File Options → Convert NIfTIs** afterwards.

---

## Usage

### Classification Modes

#### Analyze Mode

**Use case:** first-time dataset exploration, automatic `TARGET_SERIES` (config parameter) configuration, and pre-training both models with **zero manual labeling**.

**Workflow:**
1. Select directories, set Mode to **Analyze**, click **▶ Start Analysis**
2. MRID scans the DICOMs and generates series thumbnails (skipped if already done)
3. Series are clustered into visually similar groups; `ANALYZE_COVERAGE_THRESHOLD` controls how common a series type must be across the dataset to form a cluster
4. A naming window shows the 4 most representative thumbnails per cluster with a suggested name. For each cluster: edit the name, optionally tick **Prefer 3D** (see related config parameter), or untick **Include** to drop it. **Tip:** give two or more clusters the exact same name to **combine** them into a single series type (their members are merged for training, and Prefer 3D is OR-merged).
5. Click **Apply** — `TARGET_SERIES`/`PREFER_3D` are written to the project `config.py`, and high-confidence cluster members are written to `model_files/selfsupervised_classification_data.csv`
6. *(Optional)* Click **Train Model(s)** (it activates after analysis) to train the header and/or image model from that CSV — no manual labels needed
7. Switch to **Semi-Auto** or **Full-Auto**, load the models, and classify the dataset

**Technical pipeline:** frozen ResNet50 features (2048-D) → L2 normalization → PCA (≥90% variance, capped at 150 dims) → UMAP → HDBSCAN, followed by a coverage gate that keeps a cluster only if it appears in at least `ceil(ANALYZE_COVERAGE_THRESHOLD × n_exams)` **distinct exams** (clusters failing the gate and noise points are merged into the nearest surviving cluster). Suggested names are the medoid `SeriesDescription` (Levenshtein distance) among high-confidence members.

**Notes:**
- Requires `torch`, `torchvision`, `umap-learn`, and `hdbscan`
- "Other" is always appended to `TARGET_SERIES` as the catch-all label for not target series

#### Manual Mode
Review and annotate every series manually. Each study opens in a window of thumbnails with a label dropdown per series. Controls (and hotkeys) are as follows:
- **Approve (d)** save current labels and advance to the next study.
- **Skip (s)** move on to the next study without saving labels. Next time you start classification, this case will remain in the unclassified pool.
- **Previous (w)** step back to the prior study (it reloads your previously-approved labels if present). You can do this as many times as you want (taking you back to the first study), but when you next advance you will go back to the next previously unclassified study.
- **Tip:** double-click a series' label dropdown to quickly reset it to **Other** (a fast way to clear an incorrect label).

#### Semi-Auto Mode
Models pre-select the best series per target label (respecting `OVERRIDE_THRESHOLD` and `PREFER_3D`, see config parameters below); you review and adjust before saving. Predicted probabilities are logged, and selections are color-coded. Controls are the same as manual mode with the following exceptions:
- **Approve (d)** and **Skip (s)** will advance to the next study and automatically pre-select series labels based on trained models.
- **Previous (w)** steps back to the prior study, reloading your previously-approved labels if available, otherwise running the trained model-based pre-selection. Subsequent advances will take you to the next previously unclassified study.
- **Tip:** double-click a series' label dropdown to quickly reset it to **Other** (a fast way to clear an incorrect label).

#### Full-Auto Mode
Classifies and saves every study without review. Best for large datasets with consistent protocols and well-validated models.

#### Verify Mode
Quality control for already-classified studies. Thumbnails are shown in batches grouped by label; toggle any wrong ones to from **Correct** to **Incorrect** by clicking and approve. Controls are similar to manual/semi-auto mode.
- **Approve (d)** saves any **Incorrect** labels and moves on to the next batch of images.
- **Skip (s)** moves on to the next batch without saving any **Incorrect** labels. Effectively the same as approving without changing any labels to **Incorrect**.
- **Previous (w)** steps back to the prior batch, including across labels (e.g. from the first T2 batch back to the last T1 batch), and reloads the **Incorrect** marks you set earlier so you can revise them.

Study classifications are deleted only after a single confirmation at the very end of the workflow. Any studies with deleted classifications will go back into the pool of unannotated studies for subsequent manual/semi-auto mode runs. Trained models are not used or required in this mode.

**Keyboard shortcuts** (classification and verification windows):
- `a` = Stop All
- `s` = Skip
- `d` = Approve
- `w` = Previous (Manual/Semi-Auto and Verify)
- In Manual/Semi-Auto, **double-click a label dropdown** to reset it to **Other**.

### Model Types

| Model | Method                                                                                 | Strengths |
|-------|----------------------------------------------------------------------------------------|-----------|
| **DICOM Header Model** | Random Forest on configurable header fields (timing, sequence, geometry, descriptions) | Fast, interpretable, no pixel data needed |
| **Image-Based Model** | CNN on grayscale slice images (2D center slice, or 2.5D with `CNN_NUM_SLICES > 1`)     | Learns visual features, robust to metadata inconsistencies |
| **Header + Image Model** | Average of both models' output probabilities                                           | Best accuracy (recommended) |

### Training Models

You need labeled data, obtained either way:
- **Zero-label**: run [Analyze mode](#analyze-mode) — training data is collected automatically
- **Manual labeling**: classify studies with Manual or Semi-Auto mode (~50 examples per class minimum recommended; the Train/Load buttons enable once at least 10 studies are classified)

Then:
1. Click **Train Model(s)** and pick the model type (Header / Image / Both); progress is shown in a popup with a Cancel button
2. Click **Load Model(s)** in Semi-Auto or Full-Auto mode (loaded automatically after training)
3. Check `model_files/rf_model_training_summary.txt` and `cnn_model_training_summary.txt` for results

**Training tips:** balance classes, include diverse scanners/protocols, retrain as verified labels accumulate, and consider adding any additional relevant DICOM fields in the config to strengthen the header model or switching to 2.5D (`CNN_NUM_SLICES`) to strengthen the image model.

### Series Thumbnail Scrolling

Everywhere a center-slice thumbnail is shown (classification, verification, and Analyze naming windows), you can **scroll the mouse wheel over the image to page through the entire series**. Slices are loaded from the DICOM source on demand in a background thread, so thumbnail caching and window load times are unaffected. A slice counter (e.g. `12/48`) appears on the image once you start scrolling.

### Headless Batch Classification

`headless_autoclassify.py` runs Full-Auto classification from the command line (no GUI), using models trained in the app:

```bash
python headless_autoclassify.py -i /data/dicoms -o /data/output
python headless_autoclassify.py --help   # all options, including --config
```

---

## Configuration

Each project keeps its own `config.py` in the output directory. The repository default ([mrid_app/config.py](mrid_app/config.py)) is the template. At minimum, review:

- `TARGET_SERIES` — the labels you classify into
- `PREFER_3D` — which labels should prefer 3D acquisitions
- `STUDY_NAME_TEMPLATE` — output folder naming

### Config Wizard

The wizard walks you through generating or editing a project config without touching Python code:

- **New projects**: a project `config.py` is required, so when the output directory has none, starting a workflow offers **Run Config Wizard / Copy Default Config** (or Cancel to abort)
- **Existing projects**: **Configuration → Run Config Wizard** re-runs the wizard pre-filled with the active config's values

#### Pages:
1. **Output naming** (study name template, thumbnail format/quality)
2. **Classification & DICOM filtering** (target series with per-series 3D preference, ImageType include/exclude filters, stacked-series handling)
3. **Classification UI** (grid size, thresholds, ordering). An **Advanced Settings** page covers model hyperparameters, the DICOM header field list, the Analyze coverage threshold and HDBSCAN method, and the log level. Saving rewrites the config file while preserving its comments. In Analyze mode the target-series section is locked, since Analyze determines the series itself.

### Configuration Menu

| Menu item | Action |
|-----------|--------|
| **Run Config Wizard** | Guided editing of the active config (see above) |
| **Save Config File** | Copy the active config to `{output_dir}/config.py` and make it active |
| **Load Config File** | Load a config from anywhere via a file browser |
| **Edit Config File** | Open the active config in your system text editor |
| **Reload Config File** | Re-read the active config after manual edits |
| **Add Missing Parameters** | Add parameters newer MRID versions introduced to an older config, with editable suggested defaults |

When a workflow starts, a `config.py` already in the output directory is loaded automatically. Configs from older MRID versions that lack newer settings load with a warning; missing settings use defaults.

### Configuration Reference

#### Output naming

```python
STUDY_NAME_TEMPLATE = "{mrn}-{accn}"
```
Placeholders: 
- `{mrn}` (Medical Record Number)
- `{accn}` (Accession Number)
- `{study_uid}` (Study Instance UID).

The template must produce a unique name per study; invalid filename characters are replaced with hyphens.

```python
THUMBNAIL_FORMAT = 'JPEG'   # 'JPEG' or 'PNG'
THUMBNAIL_QUALITY = 75      # JPEG compression quality, 1-100
```
Affects GUI thumbnails only (the CNN preprocesses images separately).

#### Classification

```python
TARGET_SERIES = ["T1", "T2", "FLAIR", "T1c", "Other"]  # always include "Other"
```
Labels are arbitrary names (no special characters) — they do not need to match series descriptions. Analyze mode populates this automatically.

```python
PREFER_3D = {"T1": True, "T2": False, "FLAIR": True, "T1c": True, "Other": False}
# or a single bool applied to every non-Other label:
PREFER_3D = True
```
When a label prefers 3D and several series exceed `OVERRIDE_THRESHOLD` for it, a 3D acquisition (`MRAcquisitionType == '3D'`) is selected even if a 2D candidate has a higher probability. Labels missing from the dict default to `False`.

```python
INCLUDE_TYPES = ["ORIGINAL", "PRIMARY"]   # ImageType must contain one of these
EXCLUDE_TYPES = ["DERIVED", "SECONDARY", "ADC", "DIFFUSION", "LOCALIZER", "MIP", "MPR"]
IGNORE_MULTIPLE_ORIENTATIONS = True       # skip series mixing orientations
SEPARATE_STACKED_SERIES = True            # split multi-volume series
MAX_SPLIT = 5                             # max splits per stacked series
```
Use these parameters to pre-filter series before even attempting to classify.
- Primary filtering is base don the `ImageType` DICOM field, with both `INCLUDE_TYPES` and `EXCLUDE_TYPES` options.
- `IGNORE_MULTIPLE_ORIENTATIONS` excludes or include series with multiple different image orientations (e.g. 3-plane localizer images or angled axial images of the spine).
- `SEPARATE_STACKED_SERIES` optionally splits multi-volume series (e.g. DWI/B0 or multi-phase postcontrast imaging) into separate series per volume.
- `MAX_SPLIT` limits how many volumes are split off a series (when `SEPARATE_STACKED_SERIES` is True). Series exceeding this number are left intact (e.g. a stacked 60-direction DTI scan).

```python
ANALYZE_COVERAGE_THRESHOLD = 0.1
ANALYZE_HDBSCAN_CLUSTER_METHOD = 'leaf'   # 'leaf' or 'eom'
```
`ANALYZE_COVERAGE_THRESHOLD`: a cluster is valid in Analyze mode only if it covers at least this fraction of total exams in the dataset. Raise it (e.g. `0.5`) to keep only core series types; lower it (e.g. `0.05`) to surface rare sequences.

`ANALYZE_HDBSCAN_CLUSTER_METHOD`: HDBSCAN's `cluster_selection_method`. `'leaf'` (default) produces more, finer-grained clusters; `'eom'` produces fewer, larger clusters.

#### Classification UI

```python
NUM_COLUMNS = 6               # grid columns (classification & verify)
NUM_ROWS = 4                  # grid rows (verify mode batches)
THUMBNAIL_SCALE_FACTOR = 1.0  # extra multiplier for thumbnail size only
OVERRIDE_THRESHOLD = 0.5      # min probability for auto-selection (0-1)
ALLOW_DUPLICATE_LABELS = False
NUM_PARALLEL_WORKERS = 8      # workers for scanning, loading & CNN preprocessing
RANDOMIZE_STUDY_ORDER = True  # shuffle study order (Manual/Semi-Auto)
STUDY_ORDER_SEED = 42         # seed for reproducible shuffling
```

- The whole UI scales automatically with screen size; `THUMBNAIL_SCALE_FACTOR` adjusts thumbnails only (e.g. `0.75` smaller, `1.5` larger).
- `NUM_PARALLEL_WORKERS` is honored exactly (not capped to the CPU count) across every parallel stage — DICOM scanning, thumbnail generation, thumbnail loading, and CNN preprocessing.
- With `ALLOW_DUPLICATE_LABELS = False` (default), assigning a label to one series clears it from any other series in the study; set `True` to allow several series per label (useful for datasets with frequently repeated series).
- Randomized study order (using the specified seed) reduces labeling bias; Full-Auto always processes in original order.

#### CNN hyperparameters

```python
IMAGE_SIZE = 128      # input resolution (square)
NUM_EPOCHS = 100
BATCH_SIZE = 8
LR = 0.001
CNN_NUM_SLICES = 1    # 1 = 2D (center slice), >1 = 2.5D
```

**2D vs. 2.5D (`CNN_NUM_SLICES`):** by default the CNN sees only the center slice. With `CNN_NUM_SLICES = N > 1`, N evenly spaced non-edge slices are fed to the CNN as input channels — e.g. `N = 3` on a 100-slice series uses slices 25, 50, and 75 — giving the model more anatomical context at a modest cost.
- A model must be used with the same slice count it was trained with. MRID reads the slice count from saved models automatically and warns if it differs from the config; retrain to change it.
- Preprocessed training data is cached per slice count, so existing 2D caches stay valid.

#### Random Forest hyperparameters

```python
N_ESTIMATORS = 400    # number of trees
```

#### DICOM header fields

```python
DICOM_HEADER_FIELDS = [
    ("SeriesDescription", "SeriesDescription"),
    ("TR (Repetition Time)", "RepetitionTime"),
    ("TE (Echo Time)", "EchoTime"),
    # ... (display_name, pydicom_attribute_name)
]
```
The header model's input features. Field types (numerical / array / categorical / free text) are detected automatically and encoded appropriately; for example, free-text fields use semantic sentence embeddings. When you add fields, the next training run offers to back-fill them into existing classification CSVs from the original DICOMs.

##### DICOM header field encoding parameters
```python
CATEGORICAL_UNIQUE_THRESHOLD = 50  # <= this many unique values -> categorical
TEXT_MIN_WORDS = 2                 # >= this avg word count -> text
ARRAY_SIZE_THRESHOLD = 1           # > this many numbers per value -> array
```
- Header fields with less than `CATEGORICAL_UNIQUE_THRESHOLD` unique values are treated as categorical
- Header fields  with an average word count of at least `TEXT_MIN_WORDS` are treated as free text and encoded with sentence transformers
- Header fields  with more than `ARRAY_SIZE_THRESHOLD` numbers per value are treated as arrays and flattened into multiple features (e.g. `ImageOrientationPatient`).

#### Logging

```python
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
```
Standard logging level specification. Logs are written to `{output_dir}/mrid.log` with timestamps and log levels. Set to `DEBUG` for verbose output during troubleshooting.

---

## Project Structure

```
mrid/
├── mrid_app/                  # Application package
│   ├── config.py              # Default configuration (template for projects)
│   ├── gui.py                 # Main GUI and workflows
│   ├── ui_theme.py            # Dark theme, scaling, window placement
│   ├── config_wizard.py       # Guided config setup
│   ├── config_io.py           # Comment-preserving config rewriting
│   ├── series_scroller.py     # Mouse-wheel series scrolling on thumbnails
│   ├── analyze.py             # Analyze mode: ResNet50 → PCA → UMAP → HDBSCAN
│   ├── scanner.py             # DICOM scanning and hierarchy building
│   ├── processing.py          # Classification, slice loading, NIfTI conversion
│   ├── header.py              # Random Forest header model
│   ├── cnn.py                 # CNN image model (2D / 2.5D)
│   ├── prep.py                # Header preprocessing and encoding
│   ├── field_detection.py     # Automatic field type detection
│   ├── csv_field_updater.py   # Back-fill new fields into existing CSVs
│   ├── utils.py               # Naming templates and path utilities
│   └── logging_config.py      # Logging setup
├── tests/                     # Unit tests
├── tests.py                   # Test runner (python tests.py)
├── run.py                     # GUI entry point
├── headless_autoclassify.py   # Command-line Full-Auto classification
└── requirements.txt
```

**Output directory layout** (created at runtime inside your chosen Output DIR):
```
{output_dir}/
├── config.py                        # Project config
├── hierarchy.yaml                   # DICOM hierarchy index
├── series_thumbnails/               # Center-slice thumbnails (GUI + Analyze)
├── dicom_symlinks/                  # Symlinks for unorganized input folders (if needed)
├── model_files/                     # Models, encoders, caches, training data
│   ├── rf_classifier_model.joblib
│   ├── cnn_classifier_model.pth
│   ├── label_encoder.joblib
│   ├── feature_encoder.joblib
│   ├── cnn_preprocessed_data_*.pkl  # CNN preprocessing cache
│   ├── classification_data.csv
│   └── selfsupervised_classification_data.csv   # Written by Analyze mode
└── classified_outputs/
    └── {study_id}/
        ├── {study_id}_classification-results.csv
        └── {study_id}_{series_name}.nii.gz       # If conversion enabled
```

---

## Advanced Features

### File Options Menu

- **Convert NIfTIs** — batch-convert all classified series (skips ones already converted; shows progress and a summary)
- **Export DICOMS** — copy classified series to a chosen folder organized as `{mrn}/{study_uid}/{label}/{instance_uid}.dcm`
- **Fix broken filepaths** — repair a project after its input and/or output directory was moved or renamed. Confirm the current Input/Output directories (pre-filled from the main window), then **Fix filepaths** rebases the stored paths in `hierarchy.yaml`, the `dicom_symlinks`, and the classification CSVs onto the new locations (re-pointing symlinks by re-discovering each DICOM under the new input). Any image data still missing can be **viewed** (the missing files and the folders that hold them), **located** — point to the folder that holds it and choose to copy either just the project-referenced files or the entire folder into the input directory (you're shown how many were found and warned before anything is copied) — or **removed** from the hierarchy, symlinks, and CSVs to proceed without it
- **Erase Data** — selectively delete classifications, models, hierarchy/thumbnails, and/or the project config, with user confirmation

### View Menu

- **UI Scale** — opens a dialog to set a UI scale multiplier on top of the automatic screen-based scaling. Adjust it with the slider, the **−/+** buttons (0.05 steps), or by typing an exact value; changes apply to the whole app in real time. **Save** stores the value in `.mrid_prefs.json` so it is reapplied on the next launch (**Reset** returns to 1.0×).

### Automatic Folder Organization

Input folders that mix several series in one directory are detected automatically and reorganized into per-series folders using **symlinks** under `{output_dir}/dicom_symlinks/` — original files are untouched and no disk space is duplicated. This makes PACS exports and other unstructured datasets work with dcm2niix and the rest of the pipeline.

### Compressed DICOM Support

If pydicom cannot decode a file's pixel data, python-gdcm is used automatically. JPEG, JPEG2000, JPEG-LS, RLE, and other common compressed transfer syntaxes are supported transparently.

### CNN Preprocessing Cache

Training images are preprocessed in parallel (`NUM_PARALLEL_WORKERS`) and cached in `model_files/cnn_preprocessed_data_{hash}.pkl`. The cache key covers the training file list and the slice count, so retraining on unchanged data skips preprocessing entirely.

### Persistent Directories

The last-used input/output directories are remembered between sessions in `.mrid_prefs.json` next to `run.py`.

---

## Troubleshooting

#### "dcm2niix not found"
Install dcm2niix and ensure it is in your `PATH` (see [dcm2niix](#dcm2niix)). The convert toggle checks for it when enabled.

#### "CUDA out of memory" during CNN training
Reduce `BATCH_SIZE` (e.g. 4 or 2) or `IMAGE_SIZE` in the config.

#### "No classification CSV files found"
Train requires labeled data: classify at least 10 studies with Manual/Semi-Auto mode, or run Analyze mode first.

#### Models not loading
Check that `{output_dir}/model_files/` contains `rf_classifier_model.joblib`, `label_encoder.joblib`, `feature_encoder.joblib` (header model) and/or `cnn_classifier_model.pth` (image model).

#### CNN slice-count warning when loading a model
The saved model was trained with a different `CNN_NUM_SLICES` than the current config. The model still loads and runs with its own slice count; retrain if you want the new setting.

#### Compressed DICOMs not displaying correctly
Ensure `python-gdcm` is installed (`pip install python-gdcm`, or `conda install -c conda-forge gdcm` on Windows). To diagnose a specific file:
```python
import pydicom
ds = pydicom.dcmread('path/to/file.dcm', force=True)
```

#### "Detected unorganized DICOM folder" message
Informational, not an error — files are being organized with symlinks (see [Automatic Folder Organization](#automatic-folder-organization)). On Windows, symlink creation requires Developer Mode or administrator rights.

#### Slow CNN training
Install a GPU-enabled PyTorch build (see [PyTorch GPU Setup](#pytorch-gpu-setup)), or reduce `IMAGE_SIZE` / `NUM_EPOCHS`.

#### Square corners / limited fonts on Linux
Some conda Linux Tk builds ship without the font support needed for anti-aliased rounded corners. On those, MRID automatically uses clean **square** corners, picks a readable proportional system font instead of the default monospaced one, and substitutes plain-text labels for icon glyphs (e.g. `▶`) that those fonts can't render — a normal, fully functional outcome (nothing to fix).

#### Getting help
- Run `python tests.py` to verify the installation
- Check the terminal and the log file in the output directory for detailed errors (set `LOG_LEVEL = "DEBUG"` for more)
- Open a GitHub issue with your OS, Python version, error messages, and steps to reproduce

---

## Testing

```bash
python tests.py
```
All tests should pass. The suite covers configuration, DICOM scanning and processing, slice selection (2D/2.5D), field detection, config rewriting, and UI scaling.

---

## Contributing

1. Fork the repository and create a feature branch
2. Make your changes and add tests for new functionality
3. Run the test suite (`python tests.py`)
4. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

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

## Acknowledgments

- **pydicom** — DICOM file handling
- **dcm2niix** — NIfTI conversion
- **PyTorch** — deep learning framework
- **scikit-learn** — machine learning utilities
- **sentence-transformers** — semantic text encoding
- **CustomTkinter** — modern themed UI

---

**Version:** 2.0
**Released:** June 14, 2026
