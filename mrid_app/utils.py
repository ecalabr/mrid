import os
import yaml
from collections import defaultdict
from tkinter import messagebox
from typing import Optional
import re
import logging
import hashlib

try:
    from .config import STUDY_NAME_TEMPLATE
except ImportError:
    # Fallback if config import fails
    STUDY_NAME_TEMPLATE = "{mrn}-{study_uid}"

# Get logger - use standard logging here to avoid circular import with logging_config
logger = logging.getLogger('mrid')


def get_num_parallel_workers():
    """Number of parallel workers (threads/processes) to use.

    Read live from config.NUM_PARALLEL_WORKERS so the value the user set — and
    any config loaded/reloaded at runtime — is always honored. The configured
    value is used exactly (clamped only to a minimum of 1); it is intentionally
    not capped to the CPU count, since the work is I/O-bound. Every parallel
    section of the app routes through this helper."""
    try:
        from . import config
        return max(1, int(config.NUM_PARALLEL_WORKERS))
    except Exception:
        return 1


# ============================================================================
# Output Naming Functions
# ============================================================================

def sanitize_template(template, replacement='-'):
    # First, protect valid fields by temporarily replacing them
    protected = template
    placeholders = {
        '{mrn}': '__MRN_PLACEHOLDER__',
        '{study_uid}': '__STUDY_UID_PLACEHOLDER__',
        '{accn}': '__ACCN_PLACEHOLDER__'
    }
    for field, placeholder in placeholders.items():
        protected = protected.replace(field, placeholder)
    # Remove any remaining curly braces (invalid ones)
    protected = re.sub(r'[{}]', '', protected)
    # Keep only alphanumeric characters, hyphens, underscores, and the placeholder markers
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '', protected)
    # Restore valid fields
    for field, placeholder in placeholders.items():
        sanitized = sanitized.replace(placeholder, field)
    # Check for valid format fields
    valid_fields = ['{mrn}', '{study_uid}', '{accn}']
    if not any(item in sanitized for item in valid_fields):
        logger.warning("STUDY_NAME_TEMPLATE does not contain any of the required fields. Reverting to default template.")
        return "{mrn}-{study_uid}"
    if not any(item in sanitized for item in valid_fields[1:]):
        logger.warning("STUDY_NAME_TEMPLATE has only MRN as an identifier. This could lead to non-unique study directory names and potential file collisions.")
    return sanitized


# Sanitize the STUDY_NAME_TEMPLATE to ensure it's safe for file naming
STUDY_NAME_TEMPLATE = sanitize_template(STUDY_NAME_TEMPLATE)


def _generate_deterministic_placeholder(field_name, *context_values):
    """
    Generate a deterministic placeholder value for a missing template field.
    
    Args:
        field_name: Name of the missing field ('mrn', 'study_uid', or 'accn')
        *context_values: Other available values to use as context for determinism
        
    Returns:
        str: A deterministic alphanumeric placeholder value
    """
    # Filter out None and empty values from context
    valid_context = [str(v) for v in context_values if v]
    
    # Create hash input combining field name and context
    hash_input = f"{field_name}:" + "|".join(valid_context)
    
    # Generate MD5 hash (deterministic)
    hash_obj = hashlib.md5(hash_input.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    # Take first 8 characters and make it more readable
    # Use uppercase letters and numbers for better readability
    short_hash = hash_hex[:8].upper()
    
    # Return with field prefix for clarity
    return f"{field_name.upper()}-{short_hash}"


_CLASSIFIED_OUTPUTS_SUBDIR = "classified_outputs"


def get_classified_outputs_dir(output_dir: str) -> str:
    """
    Return the path to the 'classified_outputs' subfolder inside *output_dir*.

    All per-study classification folders, CSVs, and NIfTI files are stored here
    so that they are cleanly separated from scan artefacts (hierarchy.yaml,
    series_thumbnails, dicom_symlinks, config.py, model_files).

    The directory is *not* created by this function — callers should use
    ``os.makedirs(..., exist_ok=True)`` when they need to write into it.
    """
    return os.path.join(output_dir, _CLASSIFIED_OUTPUTS_SUBDIR)


def sanitize_target_series(target_series):
    """
    Sanitize TARGET_SERIES list to ensure all values are safe for filenames.
    Allows alphanumeric characters, spaces, hyphens, and underscores.
    Spaces are permitted here (they will be replaced with underscores wherever
    labels are used as directory or file name components).
    Ensures "Other" is always included in the list.

    Args:
        target_series: List of target series labels

    Returns:
        List of sanitized target series labels with "Other" guaranteed to be present
    """
    if not isinstance(target_series, list):
        logger.warning("TARGET_SERIES is not a list. Using default ['Other'].")
        return ["Other"]

    sanitized = []
    for label in target_series:
        if not isinstance(label, str):
            logger.warning(f"TARGET_SERIES contains non-string value: {label}. Skipping.")
            continue
        # Remove all characters except alphanumerics, spaces, hyphens, and underscores
        cleaned = re.sub(r'[^a-zA-Z0-9 _\-]', '', label).strip()
        if cleaned and cleaned not in sanitized:  # Skip empty strings and duplicates
            sanitized.append(cleaned)
        elif not cleaned:
            logger.warning(f"TARGET_SERIES label '{label}' became empty after sanitization. Skipping.")

    # Ensure "Other" is always present.
    # Only warn when the input looks like a full TARGET_SERIES list (more than
    # one entry), to avoid false warnings when the function is called on a
    # single label just for sanitization purposes.
    if "Other" not in sanitized:
        if len(target_series) > 1:
            logger.warning("'Other' not found in TARGET_SERIES. Adding it.")
        sanitized.append("Other")

    return sanitized


def get_study_dir_name(mrn, study_uid, accn=None):
    """
    Generate study directory name from MRN, Study UID, and optional Accession Number.
    Missing values are replaced with deterministic placeholders.

    Args:
        mrn: Patient MRN/PatientID (can be None or empty)
        study_uid: Study UID (can be None or empty)
        accn: Optional Accession Number (can be None or empty)

    Returns:
        str: Study directory name based on STUDY_NAME_TEMPLATE
    """
    # Handle missing values with deterministic placeholders
    # Use available values as context for hash generation
    
    if not mrn or mrn == "":
        mrn = _generate_deterministic_placeholder('mrn', study_uid, accn)
    
    if not study_uid or study_uid == "":
        study_uid = _generate_deterministic_placeholder('study_uid', mrn, accn)
    
    if accn is None or accn == "":
        accn = _generate_deterministic_placeholder('accn', mrn, study_uid)
    
    return STUDY_NAME_TEMPLATE.format(mrn=mrn, study_uid=study_uid, accn=accn)


def get_study_csv_name(study_id):
    """
    Generate classification results CSV filename for a study.

    Args:
        study_id: Study identifier (typically the study directory name)

    Returns:
        str: CSV filename
    """
    return f"{study_id}_classification-results.csv"


def get_nifti_name(study_id, label):
    """
    Generate NIfTI filename for a classified series.

    Args:
        study_id: Study identifier (typically the study directory name)
        label: Classification label (e.g., "T1", "T2 FLAIR")

    Returns:
        str: NIfTI filename (without directory path)
    """
    safe_label = label.lower().replace(' ', '_')
    return f"{study_id}_{safe_label}.nii.gz"


def get_nifti_prefix(study_id, label):
    """
    Generate NIfTI file prefix for dcm2niix conversion.

    Args:
        study_id: Study identifier (typically the study directory name)
        label: Classification label (e.g., "T1", "T2 FLAIR")

    Returns:
        str: NIfTI prefix for dcm2niix naming
    """
    safe_label = label.lower().replace(' ', '_')
    return f"{study_id}_{safe_label}"


def get_symlink_dir_name(mrn, study_uid, series_uid, accn="", split_index=None):
    """
    Generate symlink directory name for organizing DICOM files.
    Uses STUDY_NAME_TEMPLATE to maintain consistency with study directory naming.
    Missing values are replaced with deterministic placeholders.

    Args:
        mrn: Patient MRN/PatientID (can be None or empty)
        study_uid: Study UID (can be None or empty)
        series_uid: Series UID
        accn: Optional Accession Number (default: "")
        split_index: Optional split index for multi-volume series (0-based integer)

    Returns:
        str: Symlink directory name
    """
    # Handle missing values with deterministic placeholders
    if not mrn or mrn == "":
        mrn = _generate_deterministic_placeholder('mrn', study_uid, series_uid, accn)
    
    if not study_uid or study_uid == "":
        study_uid = _generate_deterministic_placeholder('study_uid', mrn, series_uid, accn)
    
    if accn is None or accn == "":
        accn = _generate_deterministic_placeholder('accn', mrn, study_uid, series_uid)
    
    # Use the same template as study directories for consistency
    base_name = STUDY_NAME_TEMPLATE.format(mrn=mrn, study_uid=study_uid, accn=accn)

    # Append series UID to make it unique per series
    dir_name = f"{base_name}-{series_uid}"

    # Append split index if provided
    if split_index is not None:
        dir_name = f"{dir_name}-{split_index:02d}"

    return dir_name


def extract_study_id_from_path(study_path):
    """
    Extract study ID from study directory path.

    Args:
        study_path: Full or relative path to study directory

    Returns:
        str: Study ID (directory basename)
    """
    return os.path.basename(study_path.rstrip(os.sep))



def convert_defaultdict_to_dict(obj):
    if isinstance(obj, defaultdict):
        obj = {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        obj = {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj = [convert_defaultdict_to_dict(v) for v in obj]
    return obj


def safe_get_dicom_attribute(ds, tag_name):
    """
    Safely extract DICOM attribute value from a dataset.

    Args:
        ds: pydicom Dataset object
        tag_name: DICOM attribute name as string

    Returns:
        Attribute value (converted to string tuple if MultiValue), or "None" if not found
    """
    import pydicom  # deferred: keeps GUI startup fast (see gui._LazyModule)
    value = getattr(ds, tag_name, None)
    if isinstance(value, pydicom.multival.MultiValue):
        value = str(tuple(value))
    return value if value is not None else "None"


def write_yaml_hierarchy(hierarchy, output_dir):
    yaml_path = os.path.join(output_dir, "hierarchy.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(
            hierarchy,
            f,
            indent=2,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True
        )
    logger.info(f"YAML hierarchy saved to: {yaml_path}")


def warn_if_no_gpu_for_cnn(gui_parent: Optional[object] = None) -> bool:
    """
    Warn the user before loading/training a CNN only when NO acceleration
    device is available (no CUDA/ROCm GPU and no Apple MPS). When any
    accelerator is detected, proceed silently.

    Returns:
      True if it's OK to proceed (an accelerator is available or the user
      chose to continue on CPU).
      False if the operation should be cancelled (PyTorch missing or user
      chose not to continue).
    """
    try:
        import torch
    except Exception:
        # PyTorch not installed
        try:
            messagebox.showwarning(
                "PyTorch Missing",
                "PyTorch is not installed. CNN operations require PyTorch.\n"
                "Install PyTorch to use GPU acceleration or run CNNs on CPU at your own risk.",
                parent=gui_parent,
            )
        except Exception:
            logger.warning("PyTorch is not installed. CNN operations require PyTorch.")
        return False

    # CUDA covers both NVIDIA and AMD ROCm builds
    try:
        cuda_avail = torch.cuda.is_available()
    except Exception:
        cuda_avail = False
    try:
        mps_avail = torch.backends.mps.is_available()
    except Exception:
        mps_avail = False

    # An accelerator is present — proceed without warning
    if cuda_avail or mps_avail:
        return True

    # No acceleration device at all — warn that the CPU will be used
    try:
        proceed = messagebox.askyesno(
            "No Acceleration Device Detected",
            "No GPU acceleration device was detected on this machine "
            "(no CUDA/ROCm GPU and no Apple Silicon MPS).\n\n"
            "Loading or training the CNN model will use the CPU and may be "
            "significantly slower.\n\n"
            "Choose 'Yes' to continue on CPU, or 'No' to cancel.",
            parent=gui_parent,
        )
        return bool(proceed)
    except Exception:
        # Non-GUI environment or messagebox failed -> default to not proceeding
        logger.warning("No acceleration device detected. Operation cancelled by default in non-GUI context.")
        return False