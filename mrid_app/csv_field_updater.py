"""
Utility functions for managing DICOM header fields in classification CSVs.
Handles detection of missing fields and updating existing CSVs.
"""

import os
import pandas as pd
import pydicom
from .config import DICOM_HEADER_FIELDS
from .utils import safe_get_dicom_attribute, extract_study_id_from_path, get_study_csv_name
from .logging_config import get_logger

logger = get_logger()


def get_expected_field_names():
    """Get list of expected field names from configuration."""
    return [field[0] for field in DICOM_HEADER_FIELDS]


def check_missing_fields(csv_path):
    """
    Check if a classification CSV is missing any configured DICOM header fields.

    Args:
        csv_path: Path to classification CSV file

    Returns:
        list: Missing field names, or empty list if all present
    """
    try:
        df = pd.read_csv(csv_path)
        expected_fields = get_expected_field_names()
        current_fields = set(df.columns)

        # Fields that should be in CSV (excluding Classification and Path)
        required_fields = set(expected_fields)
        missing_fields = required_fields - current_fields

        return sorted(list(missing_fields))
    except Exception as e:
        logger.warning(f"Error checking fields in {csv_path}: {e}")
        return []


def scan_for_missing_fields(output_dir):
    """
    Scan all classification CSVs in output directory for missing fields.

    Args:
        output_dir: Output directory containing study folders

    Returns:
        dict: Mapping of csv_path -> list of missing field names
    """
    csv_with_missing = {}

    if not os.path.isdir(output_dir):
        return csv_with_missing

    for study_folder in os.listdir(output_dir):
        study_path = os.path.join(output_dir, study_folder)
        if os.path.isdir(study_path):
            study_id = extract_study_id_from_path(study_path)
            csv_path = os.path.join(study_path, get_study_csv_name(study_id))

            if os.path.exists(csv_path):
                missing = check_missing_fields(csv_path)
                if missing:
                    csv_with_missing[csv_path] = missing

    return csv_with_missing


def extract_dicom_field_value(dicom_path, field_name):
    """
    Extract a single DICOM field value from a DICOM file.

    Args:
        dicom_path: Path to DICOM file
        field_name: Display name of field (from DICOM_HEADER_FIELDS)

    Returns:
        str: Field value or "None" if not found
    """
    # Find the DICOM attribute name for this field
    dicom_attr = None
    for display_name, attr_name in DICOM_HEADER_FIELDS:
        if display_name == field_name:
            dicom_attr = attr_name
            break

    if dicom_attr is None:
        return "None"

    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        return safe_get_dicom_attribute(ds, dicom_attr)
    except Exception as e:
        logger.warning(f"Error reading {dicom_path}: {e}")
        return "None"


def update_csv_with_missing_fields(csv_path, missing_fields, progress_callback=None):
    """
    Update a classification CSV by extracting missing DICOM fields.

    Args:
        csv_path: Path to classification CSV file
        missing_fields: List of missing field names to extract
        progress_callback: Optional callback(current, total, message)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read existing CSV
        df = pd.read_csv(csv_path)
        total_rows = len(df)

        if 'Path' not in df.columns:
            logger.error(f"'Path' column not found in {csv_path}")
            return False

        # Initialize missing columns with "None"
        for field in missing_fields:
            df[field] = "None"

        # Extract values for each row
        for idx, row in df.iterrows():
            dicom_path = row['Path']

            # Extract each missing field
            for field in missing_fields:
                value = extract_dicom_field_value(dicom_path, field)
                df.at[idx, field] = value

            # Progress update
            if progress_callback and (idx % 10 == 0 or idx == total_rows - 1):
                progress_callback(idx + 1, total_rows,
                                f"Updating {os.path.basename(csv_path)}: {idx + 1}/{total_rows}")

        # Save updated CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Successfully updated {csv_path} with {len(missing_fields)} field(s)")
        return True

    except Exception as e:
        logger.error(f"Error updating {csv_path}: {e}")
        return False


def update_all_csvs_with_missing_fields(csv_missing_map, progress_callback=None):
    """
    Update all CSVs that have missing fields.

    Args:
        csv_missing_map: Dict mapping csv_path -> list of missing fields
        progress_callback: Optional callback(current, total, message)

    Returns:
        tuple: (success_count, failure_count)
    """
    total_csvs = len(csv_missing_map)
    success_count = 0
    failure_count = 0

    for csv_idx, (csv_path, missing_fields) in enumerate(csv_missing_map.items()):
        if progress_callback:
            progress_callback(csv_idx, total_csvs,
                            f"Processing CSV {csv_idx + 1}/{total_csvs}")

        if update_csv_with_missing_fields(csv_path, missing_fields, progress_callback):
            success_count += 1
        else:
            failure_count += 1

    return success_count, failure_count

