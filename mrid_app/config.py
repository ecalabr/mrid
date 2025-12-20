import os

# Directories (can be overridden by UI)
DATA_DIR = "/media/ecalabr/raid/justin-data/dicoms_copy"
OUT_DIR = "/media/ecalabr/raid/justin-data/dcm_parse_out"

# Classification and model settings
TARGET_SERIES = ["T1", "T2", "FLAIR", "T1c", "Other"]
NUM_COLUMNS = 6
INCLUDE_TYPES = ["ORIGINAL", "PRIMARY"]
EXCLUDE_TYPES = ["DERIVED", "SECONDARY", "ADC", "DIFFUSION", "LOCALIZER", "MIP", "MPR"]
OVERRIDE_THRESHOLD = 0.75
IMAGE_SIZE = 64
NUM_EPOCHS = 200
BATCH_SIZE = 8
LR = 0.001
N_ESTIMATORS = 400

# DICOM header feature mapping
ARRAY_COLS_MAPPING = {
    'PixelSpacing (mm)': 2,
    'ImagePositionPatient (mm)': 3,
    'ImageOrientationPatient': 6}

INITIAL_FEATURE_COLS = [
    "SeriesDescription", "TR (Repetition Time)", "TE (Echo Time)", "TI (Inversion Time)",
    "ScanningSequence", "SequenceVariant", "ScanOptions", "SequenceName",
    "AcquisitionType", "ImageType", "Rows", "Columns", "PixelSpacing (mm)",
    "SliceThickness (mm)", "ImagePositionPatient (mm)", "ImageOrientationPatient",
    "SliceLocation (mm)", "SpacingBetweenSlices (mm)",
    "ContrastBolusAgent"]

CATEGORICAL_COLS = [
    "ScanningSequence", "SequenceVariant", "ScanOptions",
    "SequenceName", "AcquisitionType", "ImageType"]

TEXT_COLS = ['SeriesDescription', 'ContrastBolusAgent']

# NOTE: The sentence-transformers model is large and should be loaded lazily.
SEMANTIC_MODEL = None