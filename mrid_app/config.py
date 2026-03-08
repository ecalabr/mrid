# ============================================================================
# Filepath configuration options (modify only at beginning of project)
# ============================================================================
# Output naming template
STUDY_NAME_TEMPLATE = "{mrn}-{accn}"
# Thumbnail generation settings
THUMBNAIL_FORMAT = 'JPEG'  # Format for generated thumbnails: 'JPEG' or 'PNG'
THUMBNAIL_QUALITY = 75  # JPEG quality (1-100, only used if THUMBNAIL_FORMAT='JPEG')

# Classification settings
TARGET_SERIES = ["T1", "T2", "FLAIR", "T1c", "Other"]
# prefer the 3D acquisition (MRAcquisitionType == '3D') even if it's not the highest probability.
PREFER_3D = {
    "T1": True,
    "T2": False,
    "FLAIR": True,
    "T1c": True,
    "Other": False
}
# DICOM ImageType filtering
INCLUDE_TYPES = ["ORIGINAL", "PRIMARY"]
EXCLUDE_TYPES = ["DERIVED", "SECONDARY", "ADC", "DIFFUSION", "LOCALIZER", "MIP", "MPR"]
IGNORE_MULTIPLE_ORIENTATIONS = True
SEPARATE_STACKED_SERIES = True
MAX_SPLIT = 5  # max number of splits for stacked series

# Analyze mode settings
ANALYZE_COVERAGE_THRESHOLD = 0.1  # clusters will only be considered if this proportion of exams contains a member

# Classification UI options
NUM_COLUMNS = 6  # max number of columns in classification/verification UI
NUM_ROWS = 4  # max number of rows in verification UI
THUMBNAIL_SCALE_FACTOR = 1.0  # additional static scale multiplier applied to thumbnail size only (e.g. 0.75 to shrink, 1.5 to enlarge)
OVERRIDE_THRESHOLD = 0.5  # probability threshold for auto-classification
ALLOW_DUPLICATE_LABELS = False  # allow multiple series to have the same label for manual classification
NUM_PARALLEL_WORKERS = 8  # number of parallel threads for various file loading operations
RANDOMIZE_STUDY_ORDER = True  # randomize the order of studies during manual/semi-auto classification
STUDY_ORDER_SEED = 42  # seed for pseudo-random study ordering (ensures reproducibility)

# CNN hyperparameters
IMAGE_SIZE = 128
NUM_EPOCHS = 100
BATCH_SIZE = 8
LR = 0.001

# Random Forest hyperparameters
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
    ("MR Acquisition Type", "MRAcquisitionType"),

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

# ============================================================================
# Logging configuration
# ============================================================================
# Log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL
# DEBUG: Most verbose, includes subprocess commands and detailed diagnostics
# INFO: General operational messages (default)
# WARNING: Potential issues that don't prevent operation
# ERROR: Failures that may affect functionality
# CRITICAL: Severe errors that may cause program termination
LOG_LEVEL = "INFO"
