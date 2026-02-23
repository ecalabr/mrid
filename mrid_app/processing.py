import os
import subprocess
import csv
import numpy as np
import pydicom
from PIL import Image
from .header import get_series_probabilities
from .cnn import prepare_cnn_input_data
from .config import OVERRIDE_THRESHOLD, PREFER_3D, DICOM_HEADER_FIELDS
from .utils import (safe_get_dicom_attribute, get_nifti_prefix, extract_study_id_from_path,
                     get_study_dir_name, get_study_csv_name, get_classified_outputs_dir)
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
    """Backward-compatible wrapper. Supports both old signature
    background_conversion_worker(mrn, study_uid, output_root, dropdown_vars, series_dict)
    and new signature with accn parameter.
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


def load_center_slice(dicom_paths):
    """
    Load the center slice from a DICOM series.
    This is the original full implementation that scans all files.
    """
    if not dicom_paths:
        return None
    first_path = dicom_paths[0]
    base_dir = os.path.dirname(first_path)
    try:
        first_ds = pydicom.dcmread(first_path, stop_before_pixels=True, force=True)
        series_uid = first_ds.SeriesInstanceUID
    except Exception:
        return None

    series_files = []
    search_dir = base_dir
    for fname in os.listdir(search_dir):
        fpath = os.path.join(search_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            if getattr(ds, "SeriesInstanceUID", None) == series_uid:
                series_files.append(fpath)
        except Exception:
            continue

    if not series_files:
        return None

    slices = []
    ds_ref = pydicom.dcmread(series_files[0], stop_before_pixels=True, force=True)
    iop = getattr(ds_ref, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        R = np.array(iop[0:3], dtype=np.float32)
        C = np.array(iop[3:6], dtype=np.float32)
        slice_normal = np.cross(R, C)
    else:
        slice_normal = np.array([0, 0, 1], dtype=np.float32)

    for fpath in series_files:
        try:
            ds = pydicom.dcmread(fpath, force=True)
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

    slices.sort(key=lambda x: x[0])
    center_idx = len(slices) // 2
    ds_center = slices[center_idx][1]
    arr = ds_center.pixel_array.astype(np.float32)

    ww = getattr(ds_center, "WindowWidth", None)
    wc = getattr(ds_center, "WindowCenter", None)

    if ww and wc:
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = float(ww[0])
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[0])
        arr = np.clip(arr, wc - ww / 2, wc + wc / 2)

    arr -= arr.min()
    arr /= max(arr.max(), 1e-6)
    arr = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr)


def load_center_slice_fast(dicom_path):
    """
    Fast version of load_center_slice that uses cached series information.
    This version reads only the necessary files for better performance.

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


def _load_single_slice(dicom_path):
    """
    Helper to load and process a single DICOM slice into a PIL Image.

    Args:
        dicom_path: Path to DICOM file

    Returns:
        PIL Image or None
    """
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        arr = ds.pixel_array.astype(np.float32)

        # Apply windowing if available
        ww = getattr(ds, "WindowWidth", None)
        wc = getattr(ds, "WindowCenter", None)

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

        return Image.fromarray(arr)
    except Exception as e:
        logger.debug(f"Error loading slice {dicom_path}: {e}")
        return None


def load_image_tensor(dicom_path, load_center_slice_func=None, transforms_func=None):
    load_center = load_center_slice_func or load_center_slice
    transforms = transforms_func
    if transforms is None:
        try:
            from .cnn import _get_transforms
            transforms = _get_transforms()
        except Exception:
            transforms = None
    if transforms is None:
        return None
    try:
        tensor = prepare_cnn_input_data([dicom_path], load_center_slice_func=load_center)
        return tensor
    except Exception as e:
        logger.debug(f"Error applying image transform: {e}")
        return None


def get_image_probabilities(dicom_path, model, target_encoder):
    if not all([model, target_encoder]):
        return None
    input_tensor = load_image_tensor(dicom_path)
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

    # Step 1: Get predictions for all series
    temp_predictions = {}
    for series_desc, dicom_paths in study_dict.items():
        # Extract DICOM path from hierarchy entry
        first_entry = dicom_paths[0]
        if isinstance(first_entry, dict):
            first_path = first_entry.get('dicom_path', '')
        else:
            first_path = first_entry
        rf_proba = None
        cnn_proba = None

        # Get RF predictions
        if model_type in ["DICOM Header Model", "Header + Image Model"] and rf_model is not None:
            rf_proba = get_series_probabilities(first_path, rf_model, rf_feature_encoder, rf_field_types)

        # Get CNN predictions
        if model_type in ["Image-Based Model", "Header + Image Model"] and cnn_model is not None:
            cnn_proba = get_image_probabilities(first_path, cnn_model, cnn_target_encoder)

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
