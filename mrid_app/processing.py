import os
import subprocess
import csv
import time
import numpy as np
import pydicom
from PIL import Image
from .header import get_series_probabilities
from .cnn import prepare_cnn_input_data, train_cnn_model_logic
from .config import OVERRIDE_THRESHOLD


def convert_series_to_nifti(series_dir, output_path, label):
    study_id = os.path.basename(output_path)
    dcm2niix_prefix = f"{study_id}_{label.lower()}"
    target_nii_name = f"{dcm2niix_prefix}.nii.gz"
    target_json_name = f"{dcm2niix_prefix}.json"

    try:
        files_before = set(os.listdir(output_path))
    except FileNotFoundError:
        print(f"Output directory not found: {output_path}")
        return

    cmd = [
        "dcm2niix",
        "-b", "n",
        "-z", "y",
        "-f", dcm2niix_prefix,
        "-o", output_path,
        series_dir
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ dcm2niix failed with return code {result.returncode}")
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)

    files_after = set(os.listdir(output_path))
    new_files = list(files_after - files_before)

    nifti_files = sorted([
        f for f in new_files
        if f.startswith(dcm2niix_prefix) and (f.endswith('.nii') or f.endswith('.nii.gz'))
    ])

    if not nifti_files:
        print(f"❌ No NIfTI files created by dcm2niix for prefix: {dcm2niix_prefix}")
        return

    selected_nii = nifti_files[0]
    if selected_nii.endswith('.nii.gz'):
        selected_json = selected_nii.replace('.nii.gz', '.json')
    else:
        selected_json = selected_nii.replace('.nii', '.json')

    files_to_keep = {selected_nii, selected_json}

    if selected_nii != target_nii_name:
        try:
            os.rename(os.path.join(output_path, selected_nii), os.path.join(output_path, target_nii_name))
        except OSError as e:
            print(f"❌ Error renaming NIfTI file {selected_nii}: {e}")
            return

    old_json_path = os.path.join(output_path, selected_json)
    if os.path.exists(old_json_path) and selected_json != target_json_name:
        try:
            os.rename(old_json_path, os.path.join(output_path, target_json_name))
        except OSError as e:
            print(f"❌ Error renaming JSON file {selected_json}: {e}")

    for f in new_files:
        if f.startswith(dcm2niix_prefix) and f not in {target_nii_name, target_json_name}:
            try:
                if os.path.exists(os.path.join(output_path, f)):
                    os.remove(os.path.join(output_path, f))
                    print(f"Removed extra file: {f}")
            except OSError as e:
                print(f"Warning: Could not remove extra file {f}: {e}")


def background_conversion_worker(*args, convert=True, **kwargs):
    """Backward-compatible wrapper. Supports both old signature
    background_conversion_worker(mrn, study_uid, output_root, dropdown_vars, series_dict)
    and new signature with convert flag as positional or keyword.
    """
    # Unpack positional args
    mrn = None
    study_uid = None
    output_root = None
    dropdown_vars = None
    series_dict = None

    if len(args) >= 5:
        mrn, study_uid, output_root, dropdown_vars, series_dict = args[:5]
        if len(args) >= 6:
            convert = args[5]
    else:
        # Attempt to get values from kwargs (unlikely in current codepaths)
        mrn = kwargs.get('mrn')
        study_uid = kwargs.get('study_uid')
        output_root = kwargs.get('output_root')
        dropdown_vars = kwargs.get('dropdown_vars')
        series_dict = kwargs.get('series_dict')

    # Validate required parameters
    if any(v is None for v in (mrn, study_uid, output_root, dropdown_vars, series_dict)):
        raise TypeError("background_conversion_worker requires (mrn, study_uid, output_root, dropdown_vars, series_dict) as the first five arguments")

    study_out_dir = os.path.join(output_root, f"{mrn}-{study_uid}")
    os.makedirs(study_out_dir, exist_ok=True)
    csv_path = os.path.join(study_out_dir, f"{mrn}-{study_uid}_classification-results.csv")

    def safe_get(ds, tag_name):
        value = getattr(ds, tag_name, None)
        if isinstance(value, pydicom.multival.MultiValue):
            value = str(tuple(value))
        return value if value is not None else "None"

    try:
        fieldnames = [
            "Classification", "Path", "SeriesDescription", "TR (Repetition Time)",
            "TE (Echo Time)", "TI (Inversion Time)", "ScanningSequence",
            "SequenceVariant", "ScanOptions", "SequenceName", "AcquisitionType",
            "ImageType", "Rows", "Columns", "PixelSpacing (mm)",
            "SliceThickness (mm)", "ImagePositionPatient (mm)", "ImageOrientationPatient",
            "SliceLocation (mm)", "SpacingBetweenSlices (mm)",
            "ContrastBolusAgent"
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for series_desc, dicom_paths in series_dict.items():
                first_path = dicom_paths[0]
                label_val = dropdown_vars.get(first_path)
                if hasattr(label_val, 'get'):
                    label = label_val.get()
                elif isinstance(label_val, str):
                    label = label_val
                else:
                    label = "Other"
                ds = pydicom.dcmread(first_path, force=True)
                attributes_dict = {
                    "Classification": label,
                    "Path": first_path,
                    "SeriesDescription": safe_get(ds, "SeriesDescription"),
                    "TR (Repetition Time)": safe_get(ds, "RepetitionTime"),
                    "TE (Echo Time)": safe_get(ds, "EchoTime"),
                    "TI (Inversion Time)": safe_get(ds, "InversionTime"),
                    "ScanningSequence": safe_get(ds, "ScanningSequence"),
                    "SequenceVariant": safe_get(ds, "SequenceVariant"),
                    "ScanOptions": safe_get(ds, "ScanOptions"),
                    "SequenceName": safe_get(ds, "SequenceName"),
                    "AcquisitionType": safe_get(ds, "AcquisitionType"),
                    "ImageType": safe_get(ds, "ImageType"),
                    "Rows": safe_get(ds, "Rows"),
                    "Columns": safe_get(ds, "Columns"),
                    "PixelSpacing (mm)": safe_get(ds, "PixelSpacing"),
                    "SliceThickness (mm)": safe_get(ds, "SliceThickness"),
                    "ImagePositionPatient (mm)": safe_get(ds, "ImagePositionPatient"),
                    "ImageOrientationPatient": safe_get(ds, "ImageOrientationPatient"),
                    "SliceLocation (mm)": safe_get(ds, "SliceLocation"),
                    "SpacingBetweenSlices (mm)": safe_get(ds, "SpacingBetweenSlices"),
                    "ContrastBolusAgent": safe_get(ds, "ContrastBolusAgent")
                }
                row = [attributes_dict.get(field, "") for field in fieldnames]
                writer.writerow(row)

        print(f"Saved classification CSV for {mrn}-{study_uid} → {csv_path}")
    except Exception as e:
        print(f"❌ Failed to save CSV for {mrn}-{study_uid}: {e}")

    if convert:
        for series_desc, dicom_paths in series_dict.items():
            first_path = dicom_paths[0]
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
                print(f"❌ Failed to convert {first_path}: {e}")


def load_center_slice(dicom_paths):
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
        print(f"Error applying image transform: {e}")
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
        print(f"CNN Prediction probability failed: {e}")
        return None


def optimize_study_selections(study_dict, model_type, rf_model, rf_target_encoder, rf_feature_encoder, cnn_model, cnn_target_encoder, rf_field_types=None):
    """
    Optimize selections for a study. Returns a tuple:
      - final_assignments: dict mapping series_start_path -> label
      - predictions_by_path: dict mapping series_start_path -> {label: probability}

    This provides probabilities for GUIs to display. Backward callers expecting
    only final_assignments should be updated to unpack the tuple.

    Args:
        rf_field_types: Optional field types dictionary for consistent RF preprocessing
    """
    target_encoder = rf_target_encoder if rf_target_encoder is not None else cnn_target_encoder
    if target_encoder is None:
        # No encoder available; fallback to 'Other' for first path of each series
        simple = {list(paths)[0]: "Other" for paths in study_dict.values()}
        return simple, {}

    all_labels = target_encoder.classes_
    target_labels = [label for label in all_labels if label.lower() != 'other']

    final_assignments = {}
    # temporary predictions keyed by series description -> {'path':..., 'proba': np.array}
    temp_predictions = {}

    for series_desc, dicom_paths in study_dict.items():
        first_path = dicom_paths[0]
        rf_proba = None
        cnn_proba = None
        if model_type in ["DICOM Header Model", "Header + Image Model"] and rf_model is not None:
            rf_proba = get_series_probabilities(first_path, rf_model, rf_feature_encoder, rf_field_types)
        if model_type in ["Image-Based Model", "Header + Image Model"] and cnn_model is not None:
            cnn_proba = get_image_probabilities(first_path, cnn_model, cnn_target_encoder)
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
        else:
            final_assignments[first_path] = "Other"

    series_used = set(final_assignments.keys())
    best_series_for_label = {}

    for label in target_labels:
        label_index = np.where(all_labels == label)[0]
        if len(label_index) == 0:
            continue
        idx = label_index[0]
        best_proba = -1.0
        best_path = None
        for series_desc, pred in temp_predictions.items():
            current_path = pred['path']
            if current_path not in series_used:
                current_proba = pred['proba'][idx]
                if current_proba > best_proba and current_proba > OVERRIDE_THRESHOLD:
                    best_proba = current_proba
                    best_path = current_path
        if best_path:
            best_series_for_label[label] = (best_path, best_proba)

    sorted_assignments = []
    for label, (path, proba) in best_series_for_label.items():
        sorted_assignments.append((proba, path, label))
    sorted_assignments.sort(key=lambda x: x[0], reverse=True)
    for proba, path, label in sorted_assignments:
        if path not in series_used:
            final_assignments[path] = label
            series_used.add(path)
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


def full_auto_classify_study(study_dict, mrn, study_uid, output_root,
                             model_type, rf_model, rf_target_encoder, rf_feature_encoder, cnn_model,
                             cnn_target_encoder, rf_field_types=None, convert=True):
    final_assignments, predictions_by_path = optimize_study_selections(
        study_dict,
        model_type,
        rf_model, rf_target_encoder, rf_feature_encoder,
        cnn_model, cnn_target_encoder,
        rf_field_types
    )
    for path, label in final_assignments.items():
        print(f"  -> Series starting with '{path}' auto-selected as: {label}")

    background_conversion_worker(
        mrn,
        study_uid,
        output_root,
        final_assignments,
        study_dict,
        convert=convert
    )
