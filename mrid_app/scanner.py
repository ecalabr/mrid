from collections import defaultdict
from operator import itemgetter
from itertools import groupby
import os
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import (IGNORE_MULTIPLE_ORIENTATIONS, SEPARATE_STACKED_SERIES, MAX_SPLIT,
                     NUM_PARALLEL_WORKERS, THUMBNAIL_FORMAT, THUMBNAIL_QUALITY)
from .utils import get_symlink_dir_name


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
                    Tag(0x0008, 0x0050),  # AccessionNumber
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
                    "accession_number": getattr(ds, "AccessionNumber", ""),
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


def extract_and_save_center_slice(file_records, thumbnail_dir, mrn, study_uid, series_uid, split_index=None):
    """
    Extract center slice from a DICOM series and save as thumbnail image.

    Args:
        file_records: List of file record dicts from dicom_records (containing 'path', 'image_position', etc.)
        thumbnail_dir: Base directory for saving thumbnails
        mrn: Patient MRN/ID
        study_uid: Study UID
        series_uid: Series UID
        split_index: Optional split index for multi-volume series

    Returns:
        Path to saved thumbnail image, or None if extraction failed
    """
    if not file_records:
        return None

    try:
        import pydicom
        from PIL import Image

        # Build list of file paths from records
        file_paths = [rec['path'] for rec in file_records]

        # Get series UID from first file to verify consistency
        try:
            first_ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
            target_series_uid = first_ds.SeriesInstanceUID
        except Exception:
            return None

        # Load all slices with position information for sorting
        slices = []

        # Get orientation from first file for spatial sorting
        try:
            ds_ref = pydicom.dcmread(file_paths[0], stop_before_pixels=True, force=True)
            iop = getattr(ds_ref, "ImageOrientationPatient", None)
            if iop is not None and len(iop) == 6:
                R = np.array(iop[0:3], dtype=np.float32)
                C = np.array(iop[3:6], dtype=np.float32)
                slice_normal = np.cross(R, C)
            else:
                slice_normal = np.array([0, 0, 1], dtype=np.float32)
        except Exception:
            slice_normal = np.array([0, 0, 1], dtype=np.float32)

        # Load and sort all slices
        for fpath in file_paths:
            try:
                ds = pydicom.dcmread(fpath, force=True)

                # Verify this file belongs to the target series
                if getattr(ds, "SeriesInstanceUID", None) != target_series_uid:
                    continue

                # Calculate sort key based on position
                ipp = getattr(ds, "ImagePositionPatient", None)
                if ipp is not None and len(ipp) == 3:
                    ipp_vec = np.array(ipp, dtype=np.float32)
                    sort_key = np.dot(ipp_vec, slice_normal)
                else:
                    sort_key = float(getattr(ds, "InstanceNumber", 0))

                slices.append((sort_key, ds))
            except Exception:
                continue

        if not slices:
            return None

        # Sort and get center slice
        slices.sort(key=lambda x: x[0])
        center_idx = len(slices) // 2
        ds_center = slices[center_idx][1]

        # Extract pixel array
        arr = ds_center.pixel_array.astype(np.float32)

        # Apply windowing if available
        ww = getattr(ds_center, "WindowWidth", None)
        wc = getattr(ds_center, "WindowCenter", None)

        if ww and wc:
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = float(ww[0])
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = float(wc[0])
            arr = np.clip(arr, wc - ww / 2, wc + ww / 2)

        # Normalize to 0-255
        arr -= arr.min()
        arr /= max(arr.max(), 1e-6)
        arr = (arr * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_img = Image.fromarray(arr)

        # Resize to standard thumbnail size (256x256)
        pil_img = pil_img.resize((256, 256))

        # Determine file extension based on format
        file_extension = '.jpg' if THUMBNAIL_FORMAT.upper() == 'JPEG' else '.png'

        # Create thumbnail filename
        if split_index is not None:
            thumbnail_filename = f"{mrn}_{study_uid}_{series_uid}_{split_index:02d}{file_extension}"
        else:
            thumbnail_filename = f"{mrn}_{study_uid}_{series_uid}{file_extension}"

        thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)

        # Save with appropriate format and quality
        if THUMBNAIL_FORMAT.upper() == 'JPEG':
            pil_img.save(thumbnail_path, 'JPEG', quality=THUMBNAIL_QUALITY)
        else:
            pil_img.save(thumbnail_path, 'PNG')

        return thumbnail_path

    except Exception as e:
        # Silent failure - GUI will show placeholder
        return None


def build_hierarchy(dicom_records, output_dir=None, progress_callback=None, thumbnail_progress_callback=None):
    # Sort globally once
    dicom_records.sort(key=itemgetter(
        'patient', 'study_uid', 'series_number', 'series_uid', 'path'
    ))

    hierarchy = {}

    # Pre-create temp directory if needed
    temp_base = None
    thumbnail_base = None
    if output_dir:
        temp_base = os.path.join(output_dir, 'dicom_symlinks')
        if os.path.isdir(temp_base):
            shutil.rmtree(temp_base)
        os.makedirs(temp_base, exist_ok=True)

        # Create series_thumbnails directory
        thumbnail_base = os.path.join(output_dir, 'series_thumbnails')
        if os.path.isdir(thumbnail_base):
            shutil.rmtree(thumbnail_base)
        os.makedirs(thumbnail_base, exist_ok=True)

    # Check if folder is organized
    folder_organized = is_folder_organized(dicom_records)
    if not folder_organized and temp_base:
        print("Detected unorganized DICOM folder. Organizing files with symlinks in dicom_symlinks directory...")

    group_key = itemgetter('patient', 'study_uid', 'series_uid')

    # Use a counter to report progress back to caller
    processed_groups = 0

    # Collect thumbnail generation tasks for later parallel processing
    thumbnail_tasks = []

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

        # Initialize study data if this is the first time we're seeing this study
        if study_uid not in patient_dict:
            patient_dict[study_uid] = {
                "accn": first_rec.get('accession_number', '') or '',
                "series": {}
            }

        study_data = patient_dict[study_uid]
        study_dict = study_data["series"]
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

                    subseries_dir = os.path.join(temp_base, get_symlink_dir_name(
                        mrn, study_uid, series_uid, study_data["accn"], n
                    ))
                    os.makedirs(subseries_dir, exist_ok=True)

                    first_link = None
                    for item in s:
                        link_path = os.path.join(subseries_dir, os.path.basename(item['path']))
                        if not os.path.exists(link_path):
                            os.symlink(item['path'], link_path)
                        if not first_link:
                            first_link = link_path

                    # Add to hierarchy with suffix (thumbnail path will be added later)
                    desc_key = f"{base_desc} {n:02d}"
                    entry_dict = {
                        'dicom_path': first_link,
                        'thumbnail_path': None  # Will be filled in parallel phase
                    }
                    study_dict.setdefault(desc_key, []).append(entry_dict)

                    # Queue thumbnail generation task
                    if thumbnail_base and first_link:
                        thumbnail_tasks.append({
                            'files': s,
                            'thumbnail_dir': thumbnail_base,
                            'mrn': mrn,
                            'study_uid': study_uid,
                            'series_uid': series_uid,
                            'split_index': n,
                            'entry_dict': entry_dict  # Reference to update later
                        })
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
                    organized_dir = os.path.join(temp_base, get_symlink_dir_name(
                        mrn, study_uid, series_uid, study_data["accn"]
                    ))
                    os.makedirs(organized_dir, exist_ok=True)

                    first_link = None
                    for item in files:
                        link_path = os.path.join(organized_dir, os.path.basename(item['path']))
                        if not os.path.exists(link_path):
                            os.symlink(item['path'], link_path)
                        if not first_link:
                            first_link = link_path

                    # Use symlink path in hierarchy (thumbnail path will be added later)
                    entry_dict = {
                        'dicom_path': first_link,
                        'thumbnail_path': None  # Will be filled in parallel phase
                    }
                    study_dict.setdefault(base_desc, []).append(entry_dict)

                    # Queue thumbnail generation task
                    if thumbnail_base and first_link:
                        thumbnail_tasks.append({
                            'files': files,
                            'thumbnail_dir': thumbnail_base,
                            'mrn': mrn,
                            'study_uid': study_uid,
                            'series_uid': series_uid,
                            'split_index': None,
                            'entry_dict': entry_dict  # Reference to update later
                        })
                except Exception as e:
                    print(f"Error organizing series {base_desc} for patient {mrn}: {e}")
                    # Fall back to original path if symlink creation fails
                    entry_dict = {
                        'dicom_path': first_rec['path'],
                        'thumbnail_path': None  # Will be filled in parallel phase
                    }
                    study_dict.setdefault(base_desc, []).append(entry_dict)

                    # Queue thumbnail generation task
                    if thumbnail_base:
                        thumbnail_tasks.append({
                            'files': files,
                            'thumbnail_dir': thumbnail_base,
                            'mrn': mrn,
                            'study_uid': study_uid,
                            'series_uid': series_uid,
                            'split_index': None,
                            'entry_dict': entry_dict  # Reference to update later
                        })
            else:
                # Already organized or no temp_base - use original path
                entry_dict = {
                    'dicom_path': first_rec['path'],
                    'thumbnail_path': None  # Will be filled in parallel phase
                }
                study_dict.setdefault(base_desc, []).append(entry_dict)

                # Queue thumbnail generation task
                if thumbnail_base:
                    thumbnail_tasks.append({
                        'files': files,
                        'thumbnail_dir': thumbnail_base,
                        'mrn': mrn,
                        'study_uid': study_uid,
                        'series_uid': series_uid,
                        'split_index': None,
                        'entry_dict': entry_dict  # Reference to update later
                    })

        # Notify progress for this group
        try:
            if progress_callback:
                progress_callback(processed_groups)
        except Exception:
            pass

    # =========================================================================
    # PHASE 2: Generate thumbnails in parallel
    # =========================================================================
    if thumbnail_tasks and thumbnail_base:
        total_thumbnails = len(thumbnail_tasks)
        print(f"Generating {total_thumbnails} thumbnails in parallel...")

        def generate_thumbnail_task(task):
            """Worker function to generate a single thumbnail."""
            try:
                thumbnail_path = extract_and_save_center_slice(
                    task['files'],
                    task['thumbnail_dir'],
                    task['mrn'],
                    task['study_uid'],
                    task['series_uid'],
                    task['split_index']
                )
                return (task['entry_dict'], thumbnail_path)
            except Exception as e:
                return (task['entry_dict'], None)

        # Generate thumbnails in parallel
        completed_thumbnails = 0
        with ThreadPoolExecutor(max_workers=NUM_PARALLEL_WORKERS) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(generate_thumbnail_task, task): task
                for task in thumbnail_tasks
            }

            # Process results as they complete
            for future in as_completed(future_to_task):
                try:
                    entry_dict, thumbnail_path = future.result()
                    # Update the entry dict with the thumbnail path
                    entry_dict['thumbnail_path'] = thumbnail_path
                    completed_thumbnails += 1

                    # Report thumbnail progress separately with current/total
                    if thumbnail_progress_callback:
                        thumbnail_progress_callback(completed_thumbnails, total_thumbnails)
                except Exception as e:
                    completed_thumbnails += 1
                    if thumbnail_progress_callback:
                        thumbnail_progress_callback(completed_thumbnails, total_thumbnails)

        print(f"✓ Generated {completed_thumbnails} thumbnails")

    return hierarchy
