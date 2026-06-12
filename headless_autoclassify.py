#!/usr/bin/env python3
"""
Headless command-line interface for MRID full-auto classification.

This script performs the same classification logic as the GUI's "Full-Auto" mode,
but without any GUI components - suitable for headless servers and batch processing.

Usage:
    python headless_autoclassify.py --input /path/to/dicoms --output /path/to/output [options]

Examples:
    # Basic usage with DICOM Header Model (default)
    python headless_autoclassify.py -i /data/dicoms -o /data/output

    # Use Image-Based Model (CNN)
    python headless_autoclassify.py -i /data/dicoms -o /data/output --model-type cnn

    # Use combined Header + Image Model
    python headless_autoclassify.py -i /data/dicoms -o /data/output --model-type combined

    # Skip NIfTI conversion (classification only)
    python headless_autoclassify.py -i /data/dicoms -o /data/output --no-convert

    # Use existing hierarchy.yaml (skip scanning)
    python headless_autoclassify.py -i /data/dicoms -o /data/output --hierarchy /data/output/hierarchy.yaml

    # Use custom config file
    python headless_autoclassify.py -i /data/dicoms -o /data/output --config /path/to/custom_config.py
"""

import argparse
import multiprocessing as mp
import os
import sys
import yaml

import joblib

# Import logging utilities before other mrid_app modules so logging is
# available to everything imported later.
from mrid_app.logging_config import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MRID Headless Classification - Full-auto classification without GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing DICOM files"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for classification results and NIfTI files"
    )

    parser.add_argument(
        "-m", "--model-type",
        choices=["header", "cnn", "combined"],
        default="header",
        help="Model type: 'header' (DICOM Header Model), 'cnn' (Image-Based Model), "
             "or 'combined' (Header + Image Model). Default: header"
    )

    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip NIfTI conversion (classification CSV only)"
    )

    parser.add_argument(
        "--hierarchy",
        type=str,
        default=None,
        help="Path to existing hierarchy.yaml file (skip DICOM scanning)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.py file (default: use repository config)"
    )

    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers for DICOM scanning (default: CPU count)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def model_type_str(model_type_arg):
    """Convert CLI model type argument to internal model type string."""
    mapping = {
        "header": "DICOM Header Model",
        "cnn": "Image-Based Model",
        "combined": "Header + Image Model"
    }
    return mapping[model_type_arg]


def load_hierarchy_from_yaml(yaml_path):
    """Load hierarchy from a YAML file."""
    print(f"Loading hierarchy from: {yaml_path}")
    with open(yaml_path, "r") as f:
        hierarchy = yaml.safe_load(f)
    return hierarchy


def load_custom_config(config_path):
    """
    Load a custom config.py file and update the config module.
    
    Args:
        config_path: Path to the custom config.py file
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = get_logger()
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    try:
        from mrid_app import config
        import importlib.util
        
        # Get expected globals from the repository config
        expected_globals = [name for name in dir(config) if name.isupper() and not name.startswith('_')]
        
        # Load the custom config file
        spec = importlib.util.spec_from_file_location("custom_config", config_path)
        if spec is None or spec.loader is None:
            logger.error(f"Could not load config file: {config_path}")
            return False
        
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        
        # Check for missing required variables
        missing_vars = []
        for var_name in expected_globals:
            if not hasattr(custom_module, var_name):
                missing_vars.append(var_name)
        
        if missing_vars:
            logger.error(f"Custom config is missing required variables: {', '.join(missing_vars)}")
            return False
        
        # Update the config module with values from the custom config
        for var_name in expected_globals:
            if hasattr(custom_module, var_name):
                setattr(config, var_name, getattr(custom_module, var_name))
        
        logger.info(f"Successfully loaded custom config from: {config_path}")
        return True
        
    except SyntaxError as e:
        logger.error(f"Syntax error in config file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return False


def scan_dicom_directory(input_dir, output_dir, num_workers=None):
    """
    Scan a directory for DICOM files and build the hierarchy structure.

    This replicates the scanning logic from the GUI.
    """
    from mrid_app.scanner import check_dicom_batch, build_hierarchy
    from mrid_app.utils import write_yaml_hierarchy

    logger = get_logger()

    # Collect candidate files by extension; validity is checked afterwards
    logger.info(f"Scanning directory: {input_dir}")
    all_files = []
    for dcm_root, _, files in os.walk(input_dir):
        for f in files:
            if not f.startswith(".") and os.path.splitext(f)[1] in ["", ".dcm", ".dicom", ".ima"]:
                all_files.append(os.path.join(dcm_root, f))

    total_files = len(all_files)
    logger.info(f"Found {total_files} candidate files. Scanning for valid DICOM...")

    if total_files == 0:
        logger.warning("No DICOM files found!")
        return {}

    # Parallel processing with batched workers
    if num_workers is None:
        try:
            from mrid_app.config import NUM_PARALLEL_WORKERS
        except Exception:
            NUM_PARALLEL_WORKERS = 8
        num_workers = max(1, min(mp.cpu_count() - 1, NUM_PARALLEL_WORKERS))

    pool = mp.Pool(num_workers)
    results = []

    try:
        batch_size = 64
        chunks = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]

        processed = 0
        for batch_res in pool.imap_unordered(check_dicom_batch, chunks):
            if batch_res:
                results.extend(batch_res)
                processed += len(batch_res)
            else:
                processed += batch_size

            # Progress reporting
            if processed > total_files:
                processed = total_files
            percent = (processed / total_files) * 100
            print(f"\rScanning progress: {processed}/{total_files} ({percent:.1f}%)", end="", flush=True)

    finally:
        pool.close()
        pool.join()

    print()  # Newline after progress

    dicom_records = [r for r in results if r is not None]
    logger.info(f"Identified {len(dicom_records)} valid DICOM files.")

    if not dicom_records:
        logger.warning("No valid DICOM files found!")
        return {}

    groups = set((r['patient'], r['study_uid'], r['series_uid']) for r in dicom_records)
    total_groups = len(groups)
    logger.info(f"Found {total_groups} unique series across all studies.")

    logger.info("Building hierarchy and generating thumbnails...")
    processed_groups = [0]

    def progress_cb(n):
        processed_groups[0] = n
        percent = (n / total_groups) * 100 if total_groups > 0 else 0
        print(f"\rOrganizing series: {n}/{total_groups} ({percent:.1f}%)", end="", flush=True)

    thumbnail_count = [0, 0]

    def thumbnail_cb(current, total):
        thumbnail_count[0] = current
        thumbnail_count[1] = total
        percent = (current / total) * 100 if total > 0 else 0
        print(f"\rGenerating thumbnails: {current}/{total} ({percent:.1f}%)", end="", flush=True)

    hierarchy = build_hierarchy(
        dicom_records,
        output_dir,
        progress_callback=progress_cb,
        thumbnail_progress_callback=thumbnail_cb
    )

    print()  # Newline after progress

    try:
        write_yaml_hierarchy(hierarchy, output_dir)
    except Exception as e:
        logger.warning(f"Failed to write hierarchy.yaml: {e}")

    return hierarchy


def load_models(output_dir, model_type):
    """
    Load classification models from the output directory.

    Returns a tuple of (rf_model, rf_target_encoder, rf_feature_encoder, rf_field_types,
                        cnn_model, cnn_target_encoder)
    """
    logger = get_logger()

    rf_model = None
    rf_target_encoder = None
    rf_feature_encoder = None
    rf_field_types = None
    cnn_model = None
    cnn_target_encoder = None

    rf_needed = model_type in ["DICOM Header Model", "Header + Image Model"]
    cnn_needed = model_type in ["Image-Based Model", "Header + Image Model"]

    # Load RF model
    if rf_needed:
        model_files_dir = os.path.join(output_dir, "model_files")
        rf_model_path = os.path.join(model_files_dir, "rf_classifier_model.joblib")
        rf_target_encoder_path = os.path.join(model_files_dir, "label_encoder.joblib")
        rf_feature_encoder_path = os.path.join(model_files_dir, "feature_encoder.joblib")
        rf_field_types_path = os.path.join(model_files_dir, "field_types.joblib")

        if not os.path.exists(rf_model_path):
            logger.error(f"RF model not found at {rf_model_path}")
            logger.error("Please train the model first using the GUI or ensure the model files exist.")
            sys.exit(1)

        if not os.path.exists(rf_target_encoder_path) or not os.path.exists(rf_feature_encoder_path):
            logger.error("RF encoder files not found.")
            sys.exit(1)

        logger.info("Loading DICOM Header Model (Random Forest)...")
        rf_model = joblib.load(rf_model_path)
        rf_target_encoder = joblib.load(rf_target_encoder_path)
        rf_feature_encoder = joblib.load(rf_feature_encoder_path)

        if os.path.exists(rf_field_types_path):
            rf_field_types = joblib.load(rf_field_types_path)
            logger.info("  Loaded field types for consistent preprocessing")
        else:
            logger.info("  Field types not found - preprocessing will use automatic type detection")

    # Load CNN model
    if cnn_needed:
        model_files_dir = os.path.join(output_dir, "model_files")
        cnn_model_path = os.path.join(model_files_dir, "cnn_classifier_model.pth")
        cnn_target_encoder_path = os.path.join(model_files_dir, "label_encoder.joblib")

        if not os.path.exists(cnn_model_path):
            logger.error(f"CNN model not found at {cnn_model_path}")
            logger.error("Please train the model first using the GUI or ensure the model files exist.")
            sys.exit(1)

        try:
            import torch
            from mrid_app.cnn import load_cnn_model, get_torch_device

            logger.info("Loading Image-Based Model (CNN)...")

            # Best available device: CUDA (NVIDIA/AMD ROCm) > Apple MPS > CPU
            map_loc = get_torch_device()
            if map_loc is not None and map_loc.type == 'cpu':
                logger.warning("  No GPU acceleration detected, using CPU (may be slow)")
            else:
                logger.info(f"  Using device: {map_loc}")

            cnn_target_encoder = joblib.load(cnn_target_encoder_path)
            num_classes = len(cnn_target_encoder.classes_)
            cnn_model = load_cnn_model(cnn_model_path, num_classes, map_location=map_loc)

        except ImportError:
            logger.error("PyTorch is not installed. Cannot use CNN model.")
            logger.error("Install PyTorch: pip install torch torchvision")
            sys.exit(1)

    return rf_model, rf_target_encoder, rf_feature_encoder, rf_field_types, cnn_model, cnn_target_encoder


def classify_all_studies(hierarchy, output_dir, model_type,
                         rf_model, rf_target_encoder, rf_feature_encoder, rf_field_types,
                         cnn_model, cnn_target_encoder, convert=True, verbose=False):
    """
    Classify all studies in the hierarchy using full-auto mode.

    This replicates the GUI's full-auto classification loop.
    """
    from mrid_app import processing
    from mrid_app.utils import get_study_dir_name, get_study_csv_name, get_classified_outputs_dir

    logger = get_logger()

    all_studies = []
    for mrn, studies in hierarchy.items():
        for study_uid, study_data in studies.items():
            accn = study_data.get("accn", "")
            series_dict = study_data.get("series", {})
            all_studies.append((mrn, study_uid, accn, series_dict))

    total_studies = len(all_studies)
    logger.info(f"Classifying {total_studies} studies using {model_type}...")

    classified = 0
    skipped = 0

    for i, (mrn, study_uid, accn, series_dict) in enumerate(all_studies):
        study_dir_name = get_study_dir_name(mrn, study_uid, accn)

        # Check if already classified
        classification_csv = os.path.join(
            get_classified_outputs_dir(output_dir),
            study_dir_name,
            get_study_csv_name(study_dir_name)
        )
        if os.path.exists(classification_csv):
            if verbose:
                logger.debug(f"  [{i + 1}/{total_studies}] Skipping {study_dir_name} (already classified)")
            skipped += 1
            continue

        logger.info(f"  [{i + 1}/{total_studies}] Classifying: {study_dir_name}")

        processing.full_auto_classify_study(
            series_dict, mrn, study_uid, output_dir,
            model_type, rf_model, rf_target_encoder, rf_feature_encoder,
            cnn_model, cnn_target_encoder,
            rf_field_types,
            convert=convert,
            accn=accn
        )
        classified += 1

    logger.info(f"Classification complete:")
    logger.info(f"  - Classified: {classified} studies")
    logger.info(f"  - Skipped (already done): {skipped} studies")
    logger.info(f"  - Total: {total_studies} studies")


def main():
    """Main entry point for headless classification."""
    args = parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Set up logging (file handler in output directory), then load the custom
    # config if one was provided so config-loading messages are logged.
    if args.config:
        setup_logging(args.output)
        logger = get_logger()

        if not load_custom_config(args.config):
            logger.error("Failed to load custom config. Exiting.")
            sys.exit(1)
    else:
        setup_logging(args.output)
        logger = get_logger()

    model_type = model_type_str(args.model_type)

    logger.info("=" * 60)
    logger.info("MRID Headless Classification")
    logger.info("=" * 60)
    logger.info(f"Input:      {args.input}")
    logger.info(f"Output:     {args.output}")
    logger.info(f"Model:      {model_type}")
    logger.info(f"Convert:    {'Yes' if not args.no_convert else 'No'}")
    logger.info("=" * 60)

    if args.hierarchy and os.path.exists(args.hierarchy):
        hierarchy = load_hierarchy_from_yaml(args.hierarchy)
    else:
        hierarchy = scan_dicom_directory(args.input, args.output, args.jobs)

    if not hierarchy:
        logger.warning("No studies found to classify.")
        sys.exit(0)

    study_count = sum(len(studies) for studies in hierarchy.values())
    patient_count = len(hierarchy)
    logger.info(f"Found {study_count} studies from {patient_count} patients.")

    rf_model, rf_target_encoder, rf_feature_encoder, rf_field_types, cnn_model, cnn_target_encoder = load_models(
        args.output, model_type
    )

    classify_all_studies(
        hierarchy, args.output, model_type,
        rf_model, rf_target_encoder, rf_feature_encoder, rf_field_types,
        cnn_model, cnn_target_encoder,
        convert=not args.no_convert,
        verbose=args.verbose
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
