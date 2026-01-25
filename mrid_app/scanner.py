from collections import defaultdict
from operator import itemgetter
from itertools import groupby
import os
import shutil
from .config import IGNORE_MULTIPLE_ORIENTATIONS, SEPARATE_STACKED_SERIES, MAX_SPLIT


def check_dicom_batch(filepaths, include_types=None, exclude_types=None):
    """Process a batch of file paths and return a list of dicom metadata records (or None entries).

    This reduces per-file overhead by importing and initializing pydicom once per worker
    process and then parsing many files sequentially.

    Supports compressed DICOMs via python-gdcm.
    """
    results = []
    try:
        import pydicom
        # Try to import gdcm for compressed DICOM support
        try:
            import gdcm
            have_gdcm = True
        except Exception:
            have_gdcm = False

        try:
            from pydicom.tag import Tag
            have_tag = True
        except Exception:
            have_tag = False

        include_types = include_types or ["ORIGINAL", "PRIMARY"]
        exclude_types = exclude_types or ["DERIVED", "SECONDARY", "ADC", "DIFFUSION", "LOCALIZER", "MIP", "MPR"]

        # Precreate specific tags if supported
        specific = None
        if have_tag:
            try:
                specific = [
                    Tag(0x0008, 0x0008),  # ImageType
                    Tag(0x0010, 0x0020),  # PatientID
                    Tag(0x0020, 0x000D),  # StudyInstanceUID
                    Tag(0x0020, 0x000E),  # SeriesInstanceUID
                    Tag(0x0008, 0x103E),  # SeriesDescription
                    Tag(0x0020, 0x0011),  # SeriesNumber
                ]
                if IGNORE_MULTIPLE_ORIENTATIONS:
                    specific = specific + [Tag(0x0020, 0x0037)]  # ImageOrientationPatient
                if SEPARATE_STACKED_SERIES:
                    specific = specific + [Tag(0x0020, 0x0032)]  # ImagePositionPatient
            except Exception:
                specific = None

        for fp in filepaths:
            try:
                # cheap preamble check
                try:
                    with open(fp, 'rb') as fh:
                        fh.seek(128)
                        # Check for DICM preamble (validation only)
                        fh.read(4)
                except Exception:
                    pass

                ds = None
                try:
                    if specific is not None:
                        try:
                            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True, specific_tags=specific)
                        except TypeError:
                            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                    else:
                        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                except Exception:
                    # If pydicom fails and gdcm is available, try using gdcm for compressed DICOMs
                    if have_gdcm:
                        try:
                            reader = gdcm.ImageReader()
                            reader.SetFileName(fp)
                            if reader.Read():
                                # Successfully read with gdcm, now use pydicom with gdcm handler
                                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                            else:
                                results.append(None)
                                continue
                        except Exception:
                            results.append(None)
                            continue
                    else:
                        results.append(None)
                        continue

                img_type = getattr(ds, "ImageType", [])
                if not all([item in img_type for item in include_types]):
                    results.append(None)
                    continue
                if any([item in img_type for item in exclude_types]):
                    results.append(None)
                    continue

                rec = {
                    "path": fp,
                    "patient": getattr(ds, "PatientID", "UNKNOWN_MRN"),
                    "study_uid": str(getattr(ds, "StudyInstanceUID", None)),
                    "series_uid": str(getattr(ds, "SeriesInstanceUID", None)),
                    "series_desc": getattr(ds, "SeriesDescription", "UNKNOWN_SERIES"),
                    "series_number": int(getattr(ds, "SeriesNumber", 0)),
                    "image_position": getattr(ds, "ImagePositionPatient", None),
                    "image_orientation": getattr(ds, "ImageOrientationPatient", None),
                }
                results.append(rec)
            except Exception:
                results.append(None)
    except Exception:
        # If import fails, return a list of None placeholders matching inputs
        results = [None] * len(filepaths)
    return results


def convert_defaultdict_to_dict(obj):
    if isinstance(obj, defaultdict):
        obj = {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        obj = {k: convert_defaultdict_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj = [convert_defaultdict_to_dict(v) for v in obj]
    return obj


def is_folder_organized(dicom_records):
    """
    Checks if DICOM files are already organized in separate folders per series.

    Returns True if files are organized (each series in its own directory),
    False if files are unorganized (multiple series mixed in same directories).
    """
    if not dicom_records:
        return True

    # Group by series_uid and check if each series has files in only one directory
    from collections import defaultdict
    series_dirs = defaultdict(set)

    for rec in dicom_records:
        series_uid = rec.get('series_uid')
        file_path = rec.get('path')
        if series_uid and file_path:
            # Get the directory containing the DICOM file
            dir_path = os.path.dirname(file_path)
            series_dirs[series_uid].add(dir_path)

    # If any series has files in multiple directories, folder is organized
    # If multiple series share the same directory, folder is unorganized
    dir_to_series = defaultdict(set)
    for series_uid, dirs in series_dirs.items():
        for d in dirs:
            dir_to_series[d].add(series_uid)

    # Check if any directory contains files from multiple series
    for dir_path, series_set in dir_to_series.items():
        if len(series_set) > 1:
            return False  # Unorganized: multiple series in same folder

    return True  # Organized: each series in separate folders


def reorder_hierarchy(hierarchy: dict, truncate_desc=5) -> dict:
    """Reorders a nested dictionary hierarchy[mrn][study_uid][desc]
    alphabetically by MRN, study_uid, and description (desc). The final
    keys (desc) are truncated by removing the first n characters.
    """
    studies_sorted_and_truncated = {}

    for mrn, study_dict in hierarchy.items():
        studies_sorted_and_truncated[mrn] = {}
        for study_uid, desc_dict in study_dict.items():
            sorted_items = sorted(desc_dict.items())
            truncated_desc_dict = {
                original_desc[truncate_desc:]: value
                for original_desc, value in sorted_items
            }
            studies_sorted_and_truncated[mrn][study_uid] = truncated_desc_dict

    studies_sorted_by_uid = {
        mrn: dict(sorted(study_dict.items()))
        for mrn, study_dict in studies_sorted_and_truncated.items()
    }

    fully_sorted_hierarchy = dict(sorted(studies_sorted_by_uid.items()))
    return fully_sorted_hierarchy

def build_hierarchy(dicom_records, output_dir=None, progress_callback=None):
    # Sort globally once
    dicom_records.sort(key=itemgetter(
        'patient', 'study_uid', 'series_number', 'series_uid', 'path'
    ))

    hierarchy = {}

    # Pre-create temp directory if needed
    temp_base = None
    if output_dir:
        temp_base = os.path.join(output_dir, 'dicom_symlinks')
        if os.path.isdir(temp_base):
            shutil.rmtree(temp_base)
        os.makedirs(temp_base, exist_ok=True)

    # Check if folder is organized
    folder_organized = is_folder_organized(dicom_records)
    if not folder_organized and temp_base:
        print("Detected unorganized DICOM folder. Organizing files with symlinks in dicom_symlinks directory...")

    group_key = itemgetter('patient', 'study_uid', 'series_uid')

    # Use a counter to report progress back to caller
    processed_groups = 0

    for (mrn, study_uid, series_uid), member_files in groupby(dicom_records, key=group_key):
        processed_groups += 1
        try:
            files = list(member_files)
            first_rec = files[0]
        except Exception:
            # Still call back to advance progress if possible
            try:
                if progress_callback:
                    progress_callback(processed_groups)
            except Exception:
                pass
            continue

        # 1. VALIDATION: Orientation
        if IGNORE_MULTIPLE_ORIENTATIONS:
            unique_orientations = {tuple(f['image_orientation']) for f in files if f['image_orientation']}
            if len(unique_orientations) > 1:
                print(f"Warning: Skipping series {first_rec['series_desc']} for patient {mrn}: Multiple orientations")
                try:
                    if progress_callback:
                        progress_callback(processed_groups)
                except Exception:
                    pass
                continue

        # 2. VALIDATION & SPLITTING: Duplicate Positions
        # We handle splitting and normal assignment here
        patient_dict = hierarchy.setdefault(mrn, {})
        study_dict = patient_dict.setdefault(study_uid, {})
        base_desc = first_rec['series_desc'] or "UNKNOWN_SERIES"

        unique_positions_set = {tuple(f['image_position']) for f in files if f['image_position']}

        if SEPARATE_STACKED_SERIES and len(unique_positions_set) != len(files):
            # FAST SPLITTING: Use a occurrence counter to assign bucket IDs
            # This handles multi-echo/dynamic data in O(N)
            pos_counts = {}
            subseries_buckets = []  # List of lists

            try:
                for item in files:
                    pos = tuple(item['image_position'])
                    # Get the current "volume index" for this specific spatial coordinate
                    idx = pos_counts.get(pos, 0)
                    pos_counts[pos] = idx + 1

                    # Ensure the bucket exists
                    if idx >= len(subseries_buckets):
                        subseries_buckets.append([])
                    subseries_buckets[idx].append(item)

                # Process Buckets (Create symlinks)
                if len(subseries_buckets) > MAX_SPLIT:
                    print(f"Warning: Series {base_desc} for patient {mrn} exceeded max splits ({MAX_SPLIT}). Skipping.")
                    try:
                        if progress_callback:
                            progress_callback(processed_groups)
                    except Exception:
                        pass
                    continue
                for n, s in enumerate(subseries_buckets):
                    if not temp_base: continue

                    subseries_dir = os.path.join(temp_base, f'{mrn}-{series_uid}_{n:02d}')
                    os.makedirs(subseries_dir, exist_ok=True)

                    first_link = None
                    for item in s:
                        link_path = os.path.join(subseries_dir, os.path.basename(item['path']))
                        if not os.path.exists(link_path):
                            os.symlink(item['path'], link_path)
                        if not first_link:
                            first_link = link_path

                    # Add to hierarchy with suffix
                    desc_key = f"{base_desc} {n:02d}"
                    study_dict.setdefault(desc_key, []).append(first_link)
            except Exception as e:
                print(f"Error during splitting series {base_desc} for patient {mrn}: {e}")
                try:
                    if progress_callback:
                        progress_callback(processed_groups)
                except Exception:
                    pass
                continue

        else:
            # NORMAL CASE: No duplicates or separation disabled
            # If folder is unorganized, create organized symlinks
            if not folder_organized and temp_base:
                try:
                    # Create organized directory structure using symlinks
                    organized_dir = os.path.join(temp_base, f'{mrn}-{series_uid}')
                    os.makedirs(organized_dir, exist_ok=True)

                    first_link = None
                    for item in files:
                        link_path = os.path.join(organized_dir, os.path.basename(item['path']))
                        if not os.path.exists(link_path):
                            os.symlink(item['path'], link_path)
                        if not first_link:
                            first_link = link_path

                    # Use symlink path in hierarchy
                    study_dict.setdefault(base_desc, []).append(first_link)
                except Exception as e:
                    print(f"Error organizing series {base_desc} for patient {mrn}: {e}")
                    # Fall back to original path if symlink creation fails
                    study_dict.setdefault(base_desc, []).append(first_rec['path'])
            else:
                # Already organized or no temp_base - use original path
                study_dict.setdefault(base_desc, []).append(first_rec['path'])

        # Notify progress for this group
        try:
            if progress_callback:
                progress_callback(processed_groups)
        except Exception:
            pass

    return hierarchy
