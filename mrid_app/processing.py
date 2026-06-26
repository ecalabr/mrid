import os
import subprocess
import csv
import time
import numpy as np
import pydicom
from PIL import Image
from .header import get_series_probabilities, get_series_probabilities_batch
from .cnn import prepare_cnn_input_data
from .config import OVERRIDE_THRESHOLD, PREFER_3D, DICOM_HEADER_FIELDS
from .utils import (safe_get_dicom_attribute, get_nifti_prefix, extract_study_id_from_path,
                     get_study_dir_name, get_study_csv_name, get_classified_outputs_dir,
                     get_num_parallel_workers)
from .logging_config import get_logger

logger = get_logger()


def convert_series_to_nifti(series_dir, output_path, label):
    study_id = extract_study_id_from_path(output_path)
    dcm2niix_prefix = get_nifti_prefix(study_id, label)
    target_nii_name = f"{dcm2niix_prefix}.nii.gz"
    target_json_name = f"{dcm2niix_prefix}.json"

    try:
        files_before = set(os.listdir(output_path))
    except FileNotFoundError:
        logger.error(f"Output directory not found: {output_path}")
        return

    cmd = [
        "dcm2niix",
        "-b", "n",
        "-z", "y",
        "-f", dcm2niix_prefix,
        "-o", output_path,
        series_dir
    ]

    logger.debug(f"Running dcm2niix command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"dcm2niix failed with return code {result.returncode}")
        logger.debug(f"Stdout: {result.stdout}")
        logger.debug(f"Stderr: {result.stderr}")

    files_after = set(os.listdir(output_path))
    new_files = list(files_after - files_before)

    nifti_files = sorted([
        f for f in new_files
        if f.startswith(dcm2niix_prefix) and (f.endswith('.nii') or f.endswith('.nii.gz'))
    ])

    if not nifti_files:
        logger.error(f"No NIfTI files created by dcm2niix for prefix: {dcm2niix_prefix}")
        return

    selected_nii = nifti_files[0]
    if selected_nii.endswith('.nii.gz'):
        selected_json = selected_nii.replace('.nii.gz', '.json')
    else:
        selected_json = selected_nii.replace('.nii', '.json')


    if selected_nii != target_nii_name:
        try:
            os.rename(os.path.join(output_path, selected_nii), os.path.join(output_path, target_nii_name))
        except OSError as e:
            logger.error(f"Error renaming NIfTI file {selected_nii}: {e}")
            return

    old_json_path = os.path.join(output_path, selected_json)
    if os.path.exists(old_json_path) and selected_json != target_json_name:
        try:
            os.rename(old_json_path, os.path.join(output_path, target_json_name))
        except OSError as e:
            logger.error(f"Error renaming JSON file {selected_json}: {e}")

    for f in new_files:
        if f.startswith(dcm2niix_prefix) and f not in {target_nii_name, target_json_name}:
            try:
                if os.path.exists(os.path.join(output_path, f)):
                    os.remove(os.path.join(output_path, f))
                    logger.debug(f"Removed extra file: {f}")
            except OSError as e:
                logger.warning(f"Could not remove extra file {f}: {e}")


def background_conversion_worker(*args, convert=True, **kwargs):
    """Write the classification CSV for a study and, when convert is True,
    convert each labeled (non-'Other') series to NIfTI.

    Accepts positional args (mrn, study_uid, output_root, dropdown_vars,
    series_dict[, accn][, convert]); the sixth positional may be either the
    accn string or the convert bool, and the seventh (if present) is convert.
    """
    # Unpack positional args
    mrn = None
    study_uid = None
    output_root = None
    dropdown_vars = None
    series_dict = None
    accn = ""

    if len(args) >= 5:
        mrn, study_uid, output_root, dropdown_vars, series_dict = args[:5]
        if len(args) >= 6:
            # Check if 6th arg is convert flag (bool) or accn (str)
            if isinstance(args[5], bool):
                convert = args[5]
            else:
                accn = args[5]
        if len(args) >= 7:
            convert = args[6]
    else:
        # Attempt to get values from kwargs (unlikely in current codepaths)
        mrn = kwargs.get('mrn')
        study_uid = kwargs.get('study_uid')
        output_root = kwargs.get('output_root')
        dropdown_vars = kwargs.get('dropdown_vars')
        series_dict = kwargs.get('series_dict')
        accn = kwargs.get('accn', '')

    # Validate required parameters
    if any(v is None for v in (mrn, study_uid, output_root, dropdown_vars, series_dict)):
        raise TypeError("background_conversion_worker requires (mrn, study_uid, output_root, dropdown_vars, series_dict) as the first five arguments")


    study_out_dir = os.path.join(get_classified_outputs_dir(output_root), get_study_dir_name(mrn, study_uid, accn))
    os.makedirs(study_out_dir, exist_ok=True)
    study_id = get_study_dir_name(mrn, study_uid, accn)
    csv_path = os.path.join(study_out_dir, get_study_csv_name(study_id))

    try:
        # Build fieldnames dynamically from configuration
        # Start with Classification and Path, then add all configured DICOM header fields
        fieldnames = ["Classification", "Path"] + [display_name for display_name, _ in DICOM_HEADER_FIELDS]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for series_desc, dicom_paths in series_dict.items():
                # Extract DICOM path from hierarchy entry
                first_entry = dicom_paths[0]
                if isinstance(first_entry, dict):
                    first_path = first_entry.get('dicom_path', '')
                else:
                    first_path = first_entry

                label_val = dropdown_vars.get(first_path)
                if hasattr(label_val, 'get'):
                    label = label_val.get()
                elif isinstance(label_val, str):
                    label = label_val
                else:
                    label = "Other"
                ds = pydicom.dcmread(first_path, force=True)

                # Build attributes_dict dynamically from configuration
                attributes_dict = {
                    "Classification": label,
                    "Path": first_path
                }
                # Add all configured DICOM header fields
                for display_name, dicom_attr in DICOM_HEADER_FIELDS:
                    attributes_dict[display_name] = safe_get_dicom_attribute(ds, dicom_attr)

                row = [attributes_dict.get(field, "") for field in fieldnames]
                writer.writerow(row)

        logger.info(f"Saved classification CSV for {get_study_dir_name(mrn, study_uid, accn)} → {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV for {get_study_dir_name(mrn, study_uid, accn)}: {e}")

    if convert:
        for series_desc, dicom_paths in series_dict.items():
            # Extract DICOM path from hierarchy entry
            first_entry = dicom_paths[0]
            if isinstance(first_entry, dict):
                first_path = first_entry.get('dicom_path', '')
            else:
                first_path = first_entry

            label_val = dropdown_vars.get(first_path)
            if hasattr(label_val, 'get'):
                label = label_val.get()
            elif isinstance(label_val, str):
                label = label_val
            else:
                label = "Other"
            if label.lower() == "other" or not label:
                continue
            try:
                series_dir = os.path.dirname(first_path)
                convert_series_to_nifti(series_dir=series_dir, output_path=study_out_dir, label=label)
            except Exception as e:
                logger.error(f"Failed to convert {first_path}: {e}")


def get_sorted_series_files(dicom_paths):
    """
    Find all files belonging to the same series as dicom_paths[0] and return
    their paths sorted spatially along the slice normal (falling back to
    InstanceNumber when position information is missing).
    """
    if not dicom_paths:
        return []
    first_path = dicom_paths[0]
    base_dir = os.path.dirname(first_path)
    try:
        first_ds = pydicom.dcmread(first_path, stop_before_pixels=True, force=True)
        series_uid = first_ds.SeriesInstanceUID
    except Exception:
        return []

    series_files = []
    for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            if getattr(ds, "SeriesInstanceUID", None) == series_uid:
                series_files.append(fpath)
        except Exception:
            continue

    if not series_files:
        return []

    ds_ref = pydicom.dcmread(series_files[0], stop_before_pixels=True, force=True)
    iop = getattr(ds_ref, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        R = np.array(iop[0:3], dtype=np.float32)
        C = np.array(iop[3:6], dtype=np.float32)
        slice_normal = np.cross(R, C)
    else:
        slice_normal = np.array([0, 0, 1], dtype=np.float32)

    keyed = []
    for fpath in series_files:
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                sort_key = float(np.dot(np.array(ipp, dtype=np.float32), slice_normal))
            else:
                sort_key = float(getattr(ds, "InstanceNumber", 0))
            keyed.append((sort_key, fpath))
        except Exception:
            continue

    keyed.sort(key=lambda x: x[0])
    return [fpath for _, fpath in keyed]


def get_evenly_spaced_indices(num_total, num_slices):
    """
    Return num_slices evenly spaced indices into a sequence of length num_total:
    indices num_total*i//(num_slices+1) for i in 1..num_slices. With
    num_slices=1 this yields the center index (num_total // 2). Indices may
    repeat when num_slices approaches num_total.
    """
    num_slices = max(1, int(num_slices))
    return [num_total * (i + 1) // (num_slices + 1) for i in range(num_slices)]


def load_series_slices(dicom_paths, num_slices=1):
    """
    Load num_slices evenly spaced slices from a DICOM series as PIL Images.
    Returns a list of images in spatial order, or None on error.
    """
    sorted_files = get_sorted_series_files(dicom_paths)
    if not sorted_files:
        return None
    images = []
    for idx in get_evenly_spaced_indices(len(sorted_files), num_slices):
        img = _load_single_slice(sorted_files[idx])
        if img is None:
            return None
        images.append(img)
    return images


def load_center_slice(dicom_paths):
    """Load the center slice of a DICOM series as a PIL Image."""
    images = load_series_slices(dicom_paths, num_slices=1)
    return images[0] if images else None


def load_center_slice_fast(dicom_path):
    """
    Load the center slice of the series containing dicom_path as a PIL Image.

    Scans the file's directory reading headers only, collects files sharing
    the same SeriesInstanceUID, sorts them by InstanceNumber (no spatial
    sort), and decodes pixel data only for the center slice. Falls back to
    loading dicom_path itself if the series cannot be determined.

    Args:
        dicom_path: Path to a DICOM file from the series

    Returns:
        PIL Image of the center slice, or None on error
    """
    if not dicom_path or not os.path.exists(dicom_path):
        return None

    base_dir = os.path.dirname(dicom_path)

    try:
        # Read the reference file to get series UID
        ref_ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
        series_uid = getattr(ref_ds, "SeriesInstanceUID", None)

        if series_uid is None:
            # Fallback: just load the given file
            return _load_single_slice(dicom_path)

        # Quick scan to find all files in series
        series_files = []
        try:
            for fname in os.listdir(base_dir):
                fpath = os.path.join(base_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    # Quick header check
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                    if getattr(ds, "SeriesInstanceUID", None) == series_uid:
                        instance_num = getattr(ds, "InstanceNumber", 0)
                        series_files.append((instance_num, fpath))
                except Exception:
                    continue
        except Exception:
            # If directory scan fails, use the provided file
            return _load_single_slice(dicom_path)

        if not series_files:
            return _load_single_slice(dicom_path)

        # Sort by instance number and pick center
        series_files.sort(key=lambda x: x[0])
        center_idx = len(series_files) // 2
        center_path = series_files[center_idx][1]

        return _load_single_slice(center_path)

    except Exception as e:
        logger.debug(f"Error in load_center_slice_fast: {e}")
        return None


def compute_auto_window(arr):
    """Compute a robust window/level (center, width) for a pixel array using
    1st–99th percentiles, so the displayed contrast ignores outliers and most of
    the background. Returns (window_center, window_width) or None."""
    try:
        if arr is None:
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        lo = float(np.percentile(finite, 1.0))
        hi = float(np.percentile(finite, 99.0))
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            return None
        return ((lo + hi) / 2.0, hi - lo)
    except Exception:
        return None


def read_slice_array(dicom_path):
    """Return the raw float32 pixel array for a slice (for computing an auto
    window/level), or None on error."""
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        return ds.pixel_array.astype(np.float32)
    except Exception as e:
        logger.debug(f"Error reading slice array {dicom_path}: {e}")
        return None


def _resolve_window(arr, window, ds=None):
    """Resolve a (center, width) window for *arr*.

    window:
      - None    -> use the DICOM WindowCenter/Width if present (the default),
                   else no clipping (min/max normalize).
      - 'auto'  -> a percentile-based auto window from arr.
      - (wc,ww) -> an explicit window (e.g. one computed once for a whole series).
    """
    if window is None:
        if ds is None:
            return None
        ww = getattr(ds, "WindowWidth", None)
        wc = getattr(ds, "WindowCenter", None)
        if ww is None or wc is None:
            return None
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = ww[0]
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = wc[0]
        try:
            return (float(wc), float(ww))
        except Exception:
            return None
    if window == 'auto':
        return compute_auto_window(arr)
    try:
        wc, ww = window
        return (float(wc), float(ww))
    except Exception:
        return None


def _load_single_slice(dicom_path, window=None):
    """
    Helper to load and process a single DICOM slice into a PIL Image.

    Args:
        dicom_path: Path to DICOM file
        window: windowing to apply — None (DICOM default), 'auto', or an explicit
            (center, width) tuple. See _resolve_window.

    Returns:
        PIL Image or None
    """
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        arr = ds.pixel_array.astype(np.float32)

        win = _resolve_window(arr, window, ds=ds)
        if win is not None:
            wc, ww = win
            if ww > 0:
                arr = np.clip(arr, wc - ww / 2, wc + ww / 2)

        # Normalize to 0-255
        arr -= arr.min()
        arr /= max(arr.max(), 1e-6)
        arr = (arr * 255).astype(np.uint8)

        return Image.fromarray(arr)
    except Exception as e:
        logger.debug(f"Error loading slice {dicom_path}: {e}")
        return None


def load_image_tensor(dicom_path, num_slices=1, load_slices_func=None, image_size=None):
    load_slices = load_slices_func or load_series_slices
    try:
        tensor = prepare_cnn_input_data([dicom_path], load_slices_func=load_slices,
                                        num_slices=num_slices, image_size=image_size)
        return tensor
    except Exception as e:
        logger.debug(f"Error applying image transform: {e}")
        return None


def get_image_probabilities(dicom_path, model, target_encoder):
    if not all([model, target_encoder]):
        return None
    # Input slice count and image size are dictated by the loaded model, so
    # inference always matches how the model was trained.
    try:
        num_slices = int(model.conv1.in_channels)
    except Exception:
        num_slices = 1
    image_size = getattr(model, 'image_size', None)
    input_tensor = load_image_tensor(dicom_path, num_slices=num_slices, image_size=image_size)
    if input_tensor is None:
        return None
    try:
        import torch
        model.eval()
        with torch.no_grad():
            # Move input to the same device as the model to avoid device mismatch
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = torch.device('cpu')

            # If input tensor is on CPU, move to model_device
            try:
                input_tensor = input_tensor.to(model_device)
            except Exception:
                # Fallback: attempt to convert to appropriate dtype/device
                input_tensor = input_tensor.clone().detach().to(model_device)

            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            return probabilities
    except Exception as e:
        logger.debug(f"CNN Prediction probability failed: {e}")
        return None


def get_image_probabilities_batch(dicom_paths, model, target_encoder):
    """Predict CNN probabilities for many series at once.

    Returns a list aligned with *dicom_paths* (None where the image couldn't be
    loaded). Per-series image tensors are loaded in parallel (I/O bound) and run
    through the model as a single batched forward pass, so a study's series share
    one model call instead of one each."""
    n = len(dicom_paths)
    results = [None] * n
    if not all([model, target_encoder]) or n == 0:
        return results
    try:
        import torch
    except Exception:
        return results

    try:
        num_slices = int(model.conv1.in_channels)
    except Exception:
        num_slices = 1
    image_size = getattr(model, 'image_size', None)

    # Load image tensors in parallel — the slow part is DICOM I/O.
    tensors = [None] * n
    workers = max(1, min(get_num_parallel_workers(), n))
    def _load(i):
        return i, load_image_tensor(dicom_paths[i], num_slices=num_slices, image_size=image_size)
    _t = time.perf_counter()   # [BENCH]
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, t in ex.map(_load, range(n)):
                tensors[i] = t
    except Exception:
        for i in range(n):
            tensors[i] = load_image_tensor(dicom_paths[i], num_slices=num_slices, image_size=image_size)
    logger.info(f"[BENCH]   CNN image load ({n} series, {workers} workers, "
                f"{num_slices} slice(s)): {(time.perf_counter()-_t)*1000:.0f} ms")  # [BENCH]

    valid = [i for i, t in enumerate(tensors) if t is not None]
    if not valid:
        return results
    try:
        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        _t = time.perf_counter()   # [BENCH]
        with torch.no_grad():
            batch = torch.cat([tensors[i].to(device) for i in valid], dim=0)
            outputs = model(batch)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        logger.info(f"[BENCH]   CNN forward (batch {len(valid)}, {device}): "
                    f"{(time.perf_counter()-_t)*1000:.0f} ms")  # [BENCH]
        for j, i in enumerate(valid):
            results[i] = probs[j]
    except Exception as e:
        logger.debug(f"Batched CNN prediction failed: {e}")
    return results


def optimize_study_selections(study_dict, model_type, rf_model, rf_target_encoder, rf_feature_encoder, cnn_model, cnn_target_encoder, rf_field_types=None):
    """
    Optimize series selections for a study. Returns a tuple:
      - final_assignments: dict mapping series_start_path -> label
      - predictions_by_path: dict mapping series_start_path -> {label: probability}

    Selection logic:
      1. Get predictions for all series in the study
      2. For each target label (excluding 'Other'):
         a. Find all series where probability > OVERRIDE_THRESHOLD
         b. If PREFER_3D is enabled for this label and there's a 3D series, select it
         c. Otherwise, select the series with highest probability
      3. Assign labels in order of confidence (highest probability first)
      4. Ensure each series is assigned only once
      5. Remaining series are assigned 'Other'

    Args:
        rf_field_types: Optional field types dictionary for consistent RF preprocessing
    """
    target_encoder = rf_target_encoder if rf_target_encoder is not None else cnn_target_encoder
    if target_encoder is None:
        # No encoder available; fallback to 'Other' for first path of each series
        def _get_first_path(paths):
            first_entry = paths[0] if paths else None
            if isinstance(first_entry, dict):
                return first_entry.get('dicom_path', '')
            return first_entry or ''
        simple = {_get_first_path(paths): "Other" for paths in study_dict.values()}
        return simple, {}

    all_labels = target_encoder.classes_
    target_labels = [label for label in all_labels if label.lower() != 'other']

    # Step 1: Get predictions for all series. Inference is BATCHED across the
    # study's series — one sentence-transformer encode + one RF predict, and one
    # CNN forward pass — instead of one call per series, which dominated the time
    # to advance between exams in Semi-Auto.
    series_list = []   # (series_desc, first_path)
    for series_desc, dicom_paths in study_dict.items():
        first_entry = dicom_paths[0]
        first_path = first_entry.get('dicom_path', '') if isinstance(first_entry, dict) else first_entry
        series_list.append((series_desc, first_path))
    paths = [p for _, p in series_list]

    # ---- [BENCH] temporary inference timing (remove later) ----
    _bench_n = len(paths)
    _bench_t0 = time.perf_counter()

    rf_list = None
    if model_type in ["DICOM Header Model", "Header + Image Model"] and rf_model is not None:
        _t = time.perf_counter()
        rf_list = get_series_probabilities_batch(paths, rf_model, rf_feature_encoder, rf_field_types)
        logger.info(f"[BENCH] RF inference ({_bench_n} series): {(time.perf_counter()-_t)*1000:.0f} ms")
    cnn_list = None
    if model_type in ["Image-Based Model", "Header + Image Model"] and cnn_model is not None:
        _t = time.perf_counter()
        cnn_list = get_image_probabilities_batch(paths, cnn_model, cnn_target_encoder)
        logger.info(f"[BENCH] CNN inference ({_bench_n} series): {(time.perf_counter()-_t)*1000:.0f} ms")
    logger.info(f"[BENCH] optimize_study_selections TOTAL ({_bench_n} series, {model_type}): "
                f"{(time.perf_counter()-_bench_t0)*1000:.0f} ms")
    # ---- end [BENCH] ----

    temp_predictions = {}
    for i, (series_desc, first_path) in enumerate(series_list):
        rf_proba = rf_list[i] if rf_list is not None else None
        cnn_proba = cnn_list[i] if cnn_list is not None else None

        # Combine predictions based on model type
        final_proba = None
        if model_type == "Header + Image Model":
            if rf_proba is not None and cnn_proba is not None:
                final_proba = (rf_proba + cnn_proba) / 2
            elif rf_proba is not None:
                final_proba = rf_proba
            elif cnn_proba is not None:
                final_proba = cnn_proba
        elif model_type == "DICOM Header Model":
            final_proba = rf_proba
        elif model_type == "Image-Based Model":
            final_proba = cnn_proba

        if final_proba is not None:
            temp_predictions[series_desc] = {
                'path': first_path,
                'proba': final_proba,
                'rf_proba': rf_proba,
                'cnn_proba': cnn_proba
            }

    # Step 2: For each target label, find the best series
    label_candidates = {}  # label -> list of (path, proba, is_3d)

    for label in target_labels:
        label_index = np.where(all_labels == label)[0]
        if len(label_index) == 0:
            continue
        idx = label_index[0]

        candidates = []
        for series_desc, pred in temp_predictions.items():
            current_path = pred['path']
            current_proba = pred['proba'][idx]

            # Only consider series that exceed the threshold
            if current_proba > OVERRIDE_THRESHOLD:
                # Check if this is a 3D acquisition
                is_3d = check_is_3d_acquisition(current_path)
                candidates.append((current_path, current_proba, is_3d))

        if candidates:
            # Apply PREFER_3D logic if enabled for this label
            # PREFER_3D may be a bool (applies to all non-Other series) or a dict
            if isinstance(PREFER_3D, bool):
                prefer_3d = PREFER_3D if label != "Other" else False
            else:
                prefer_3d = PREFER_3D.get(label, False)

            if prefer_3d:
                # Check if there's at least one 3D series
                has_3d = any(is_3d for _, _, is_3d in candidates)
                if has_3d:
                    # Filter to only 3D series and pick the best one
                    candidates_3d = [c for c in candidates if c[2]]
                    best_candidate = max(candidates_3d, key=lambda x: x[1])
                else:
                    # No 3D series, pick best overall
                    best_candidate = max(candidates, key=lambda x: x[1])
            else:
                # Pick best overall
                best_candidate = max(candidates, key=lambda x: x[1])

            label_candidates[label] = best_candidate

    # Step 3: Sort candidates by probability (highest first) and assign
    # This ensures the most confident predictions get their preferred series first
    sorted_assignments = []
    for label, (path, proba, is_3d) in label_candidates.items():
        sorted_assignments.append((proba, path, label))
    sorted_assignments.sort(key=lambda x: x[0], reverse=True)

    # Step 4: Assign labels, ensuring each series is used only once
    final_assignments = {}
    series_used = set()

    for proba, path, label in sorted_assignments:
        if path not in series_used:
            final_assignments[path] = label
            series_used.add(path)

    # Step 5: Assign 'Other' to remaining series
    for series_desc, pred in temp_predictions.items():
        path = pred['path']
        if path not in series_used:
            final_assignments[path] = 'Other'

    # Build a predictions_by_path map: path -> {label: prob}
    predictions_by_path = {}
    try:
        for series_desc, pred in temp_predictions.items():
            path = pred['path']
            proba = pred.get('proba')
            rf_p = pred.get('rf_proba')
            cnn_p = pred.get('cnn_proba')
            entry = {}
            if proba is not None:
                entry['combined'] = {str(lbl): float(p) for lbl, p in zip(all_labels, proba)}
            if rf_p is not None:
                entry['rf'] = {str(lbl): float(p) for lbl, p in zip(all_labels, rf_p)}
            if cnn_p is not None:
                entry['cnn'] = {str(lbl): float(p) for lbl, p in zip(all_labels, cnn_p)}
            predictions_by_path[path] = entry
    except Exception:
        predictions_by_path = {}

    return final_assignments, predictions_by_path


def check_is_3d_acquisition(dicom_path):
    """
    Check if a DICOM series is a 3D acquisition based on MRAcquisitionType.

    Args:
        dicom_path: Path to DICOM file

    Returns:
        bool: True if MRAcquisitionType is '3D', False otherwise
    """
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
        acq_type = getattr(ds, 'MRAcquisitionType', None)
        if acq_type is not None:
            return str(acq_type).strip().upper() == '3D'
    except Exception as e:
        logger.debug(f"Error checking 3D acquisition for {dicom_path}: {e}")
    return False


def full_auto_classify_study(study_dict, mrn, study_uid, output_root,
                             model_type, rf_model, rf_target_encoder, rf_feature_encoder, cnn_model,
                             cnn_target_encoder, rf_field_types=None, convert=True, accn=""):
    final_assignments, predictions_by_path = optimize_study_selections(
        study_dict,
        model_type,
        rf_model, rf_target_encoder, rf_feature_encoder,
        cnn_model, cnn_target_encoder,
        rf_field_types
    )
    for path, label in final_assignments.items():
        logger.info(f"  -> Series starting with '{path}' auto-selected as: {label}")

    background_conversion_worker(
        mrn,
        study_uid,
        output_root,
        final_assignments,
        study_dict,
        accn,
        convert
    )
