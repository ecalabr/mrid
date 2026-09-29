# ============================================================================
# Filepath configuration options (modify only at beginning of project)
# ============================================================================
# Output naming template; valid fields are {mrn}, {study_uid}, and {accn}
STUDY_NAME_TEMPLATE = "{mrn}-{accn}"
# Thumbnail generation settings
THUMBNAIL_FORMAT = 'JPEG'  # Format for generated thumbnails: 'JPEG' or 'PNG'
THUMBNAIL_QUALITY = 75  # JPEG quality (1-100, only used if THUMBNAIL_FORMAT='JPEG')

# Classification settings
TARGET_SERIES = ["T1", "T2", "FLAIR", "T1c", "Other"]  # labels to classify series into; "Other" catches everything else
# prefer the 3D acquisition (MRAcquisitionType == '3D') even if it's not the highest probability.
PREFER_3D = {
    "T1": True,
    "T2": False,
    "FLAIR": True,
    "T1c": True,
    "Other": False
}
# DICOM ImageType filtering
INCLUDE_TYPES = ["ORIGINAL", "PRIMARY"]  # keep only images whose ImageType contains all of these
EXCLUDE_TYPES = ["DERIVED", "SECONDARY", "ADC", "DIFFUSION", "LOCALIZER", "MIP", "MPR"]  # skip images whose ImageType contains any of these
IGNORE_MULTIPLE_ORIENTATIONS = True  # skip series whose images have more than one orientation (e.g. localizers)
SEPARATE_STACKED_SERIES = True  # split series that repeat slice positions (multi-echo, dynamic) into separate series
MAX_SPLIT = 5  # max number of splits for stacked series

# Analyze mode settings
ANALYZE_COVERAGE_THRESHOLD = 0.1  # clusters will only be considered if this proportion of exams contains a member
# HDBSCAN cluster_selection_method: 'eom' (fewer, larger clusters) or 'leaf' (more, finer clusters)
ANALYZE_HDBSCAN_CLUSTER_METHOD = 'leaf'

# Classification UI options
NUM_COLUMNS = 6  # max number of columns in classification/verification UI
NUM_ROWS = 4  # max number of rows in verification UI
THUMBNAIL_SCALE_FACTOR = 1.0  # additional static scale multiplier applied to thumbnail size only (e.g. 0.75 to shrink, 1.5 to enlarge)
OVERRIDE_THRESHOLD = 0.5  # probability threshold for auto-classification
# How the header and image model probabilities are combined when using the
# "Header + Image Model" type. One of:
#   'reliability' weight each model per class by the F1 it achieved on its own
#                 held-out validation split, recorded at training time. The
#                 default: it scored highest of the fusion rules on most
#                 evaluation datasets, and by a wide margin where one model was
#                 much weaker than the other. Falls back to 'average' when those
#                 metrics are absent (models trained before MRID recorded them).
#   'average'     mean of both models (sum rule); the historical behavior
#   'product'     geometric mean sqrt(p_rf * p_cnn); requires both models to be
#                 confident, so it is the most conservative rule
#   'max'         per-class maximum (max rule); the most confident model wins
#   'confidence'  weight each model per series by its certainty (entropy based)
FUSION_METHOD = 'reliability'
ALLOW_DUPLICATE_LABELS = False  # allow multiple series to have the same label for manual classification
NUM_PARALLEL_WORKERS = 8  # number of parallel workers (threads/processes) for file scanning, loading, and CNN data loading
RANDOMIZE_STUDY_ORDER = True  # randomize the order of studies during manual/semi-auto classification
# Seed for every pseudo-random part of MRID: study ordering, the train/validation
# splits, Random Forest fitting, CNN weight initialization, batch shuffling and
# augmentation, and Analyze mode's PCA/UMAP. Keeping one seed means a project's
# results can be reproduced from its config. Training is deterministic for
# repeated runs on the same machine and GPU; results may still differ across
# different GPUs or PyTorch builds.
STUDY_ORDER_SEED = 42

# CNN hyperparameters
IMAGE_SIZE = 128  # CNN input size in pixels (images are resized to IMAGE_SIZE x IMAGE_SIZE)
NUM_EPOCHS = 100  # CNN training epochs
BATCH_SIZE = 8  # CNN training batch size
LR = 0.001  # CNN learning rate
# Number of evenly spaced slices fed to the CNN as input channels.
# 1 = classic 2D mode (center slice only). N > 1 = 2.5D mode: for a series of
# length L, slices at indices L*1//(N+1), L*2//(N+1), ..., L*N//(N+1) are used
# (e.g. N=3 on a 100-slice series -> slices 25, 50, 75).
# NOTE: a model must be used with the same CNN_NUM_SLICES it was trained with.
CNN_NUM_SLICES = 1

# Random Forest hyperparameters
N_ESTIMATORS = 400  # number of trees in the Random Forest

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

# (display name, DICOM attribute) pairs used as header-model features
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

# Sentence-transformers model that embeds free-text header fields (e.g.
# SeriesDescription) for the header model: any model name on the Hugging Face
# Hub (e.g. 'all-MiniLM-L6-v2', 'all-mpnet-base-v2') or a local model folder.
# It is downloaded on first use. A trained header model keeps using the model it
# was trained with; retrain it after changing this.
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'

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
