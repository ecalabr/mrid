import os
import yaml
from collections import defaultdict
from tkinter import messagebox
from typing import Optional
import pydicom


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
    print(f"YAML hierarchy saved to:\n{yaml_path}")


def warn_if_no_gpu_for_cnn(gui_parent: Optional[object] = None) -> bool:
    """
    Helper to warn the user before loading/training a CNN when no CUDA GPU is available.

    Returns:
      True if it's OK to proceed (GPU is available or the user chose to continue on CPU).
      False if the operation should be cancelled (PyTorch missing or user chose not to continue).
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
            print("PyTorch is not installed. CNN operations require PyTorch.")
        return False

    try:
        cuda_avail = torch.cuda.is_available()
    except Exception:
        cuda_avail = False

    if not cuda_avail:
        # Ask the user whether to proceed on CPU
        try:
            proceed = messagebox.askyesno(
                "No GPU Detected",
                "No CUDA-capable GPU was detected on this machine.\n\n"
                "Loading or training the CNN model will use the CPU and may be significantly slower.\n\n"
                "Choose 'Yes' to continue on CPU, or 'No' to cancel.",
                parent=gui_parent,
            )
            return bool(proceed)
        except Exception:
            # Non-GUI environment or messagebox failed -> default to not proceeding
            print("No CUDA-capable GPU detected. Operation cancelled by default in non-GUI context.")
            return False

    return True
