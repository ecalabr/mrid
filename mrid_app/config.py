import os

# Directories (can be overridden by UI)
DATA_DIR = "/media/ecalabr/raid/justin-data/dicoms_copy"
OUT_DIR = "/media/ecalabr/raid/justin-data/dcm_parse_out"

# Classification and model settings
TARGET_SERIES = ["T1", "T2", "FLAIR", "T1c", "Other"]
NUM_COLUMNS = 6
INCLUDE_TYPES = ["ORIGINAL", "PRIMARY"]
EXCLUDE_TYPES = ["DERIVED", "SECONDARY", "ADC", "DIFFUSION", "LOCALIZER", "MIP", "MPR"]
IGNORE_MULTIPLE_ORIENTATIONS = True
SEPARATE_STACKED_SERIES = True
MAX_SPLIT = 5  # max number of splits for stacked series
OVERRIDE_THRESHOLD = 0.5
IMAGE_SIZE = 64
NUM_EPOCHS = 200
BATCH_SIZE = 8
LR = 0.001
N_ESTIMATORS = 400

# ============================================================================
# DICOM HEADER FIELDS CONFIGURATION
# ============================================================================
# Users can specify which DICOM header fields to use for the header classifier
# The system will automatically detect the field type and encode appropriately
#
# Format: List of tuples (display_name, DICOM_attribute_name)
# - display_name: Human-readable name for the feature
# - DICOM_attribute_name: Actual DICOM tag attribute name
#
# The system automatically detects:
# - Numerical: Single numbers (TR, TE, Rows, Columns, etc.)
# - Numerical Arrays: Lists/tuples of numbers (PixelSpacing, ImagePosition, etc.)
# - Categorical: Limited set of discrete values (ScanningSequence, ImageType, etc.)
# - Text: Free-form text for semantic encoding (SeriesDescription, etc.)

DICOM_HEADER_FIELDS = [
    # Series identification and description
    ("SeriesDescription", "SeriesDescription"),

    # Timing parameters
    ("TR (Repetition Time)", "RepetitionTime"),
    ("TE (Echo Time)", "EchoTime"),
    ("TI (Inversion Time)", "InversionTime"),

    # Sequence information
    ("ScanningSequence", "ScanningSequence"),
    ("SequenceVariant", "SequenceVariant"),
    ("ScanOptions", "ScanOptions"),
    ("SequenceName", "SequenceName"),

    # Acquisition properties
    ("AcquisitionType", "AcquisitionType"),
    ("ImageType", "ImageType"),
    ("MR AcquisitionType", "MRAcquisitionType"),

    # Image dimensions
    ("Rows", "Rows"),
    ("Columns", "Columns"),

    # Spatial information (arrays)
    ("PixelSpacing (mm)", "PixelSpacing"),
    ("SliceThickness (mm)", "SliceThickness"),
    ("ImagePositionPatient (mm)", "ImagePositionPatient"),
    ("ImageOrientationPatient", "ImageOrientationPatient"),
    ("SliceLocation (mm)", "SliceLocation"),
    ("SpacingBetweenSlices (mm)", "SpacingBetweenSlices"),

    # Contrast information
    ("ContrastBolusAgent", "ContrastBolusAgent"),
]

# ============================================================================
# ADVANCED CONFIGURATION (Usually no need to modify)
# ============================================================================

# Thresholds for automatic type detection
CATEGORICAL_UNIQUE_THRESHOLD = 50  # Fields with <= this many unique values are categorical
TEXT_MIN_WORDS = 2  # Fields with avg >= this many words are treated as text
ARRAY_SIZE_THRESHOLD = 1  # Fields with values containing > 1 number are arrays

# NOTE: The sentence-transformers model is large and should be loaded lazily.
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'
SEMANTIC_MODEL = None  # always None initially
