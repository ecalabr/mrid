import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pydicom
from .prep import _preprocess_df, encode_text_features
from .config import N_ESTIMATORS, DICOM_HEADER_FIELDS
from .utils import safe_get_dicom_attribute, extract_study_id_from_path, get_study_csv_name, get_classified_outputs_dir, filter_nonempty_frames, get_num_parallel_workers
from .csv_field_updater import (
    scan_for_missing_fields,
    update_all_csvs_with_missing_fields
)
from .logging_config import get_logger

logger = get_logger()


def TrainHeaderClassifierLogic(output_dir, gui, update_missing_fields_callback=None, preloaded_df=None):
    """
    Train header classifier with optional missing field update.

    Args:
        output_dir: Output directory containing classification CSVs
        gui: Training GUI for status updates
        update_missing_fields_callback: Optional callback that returns True if user wants to update CSVs
        preloaded_df: Optional pre-loaded DataFrame to use instead of aggregating per-study CSVs.
                      When provided, CSV scanning/aggregation is skipped entirely.
    """
    from .logging_config import update_log_file_location

    # Set up file logging now that we have the output directory
    os.makedirs(output_dir, exist_ok=True)
    update_log_file_location(output_dir)

    if not all([pd, RandomForestClassifier, joblib, LabelEncoder, OrdinalEncoder]):
        gui.update_status("❌ Error: Missing required dependencies for Header Model.")
        time.sleep(2)
        gui.close()
        return

    try:
        if preloaded_df is not None:
            # ── Analyze mode: use the pre-loaded self-supervised DataFrame ────
            gui.update_status("1/6: Using self-supervised classification data for Header Model…")
            master_df = preloaded_df.copy()
            model_files_dir = os.path.join(output_dir, "model_files")
            os.makedirs(model_files_dir, exist_ok=True)
        else:
            # ── Normal mode: check for missing fields then aggregate CSVs ─────
            gui.update_status("Checking for missing DICOM header fields...")
            if gui.is_cancelled():
                logger.info("Header model training cancelled by user.")
                gui.close()
                return
            classified_dir = get_classified_outputs_dir(output_dir)
            scan_root = classified_dir if os.path.isdir(classified_dir) else output_dir
            csv_missing_map = scan_for_missing_fields(scan_root)

            if csv_missing_map:
                # Found CSVs with missing fields
                total_csvs = len(csv_missing_map)
                all_missing_fields = set()
                for missing_fields in csv_missing_map.values():
                    all_missing_fields.update(missing_fields)

                missing_fields_str = ", ".join(sorted(all_missing_fields))

                # Prompt user if callback provided
                should_update = False
                if update_missing_fields_callback:
                    should_update = update_missing_fields_callback(
                        total_csvs,
                        list(all_missing_fields)
                    )

                if should_update:
                    # Update CSVs with missing fields
                    gui.update_status(f"Updating {total_csvs} CSV(s) with missing fields...")

                    def update_progress(current, total, message):
                        gui.update_status(f"Updating CSVs: {message}")

                    success_count, failure_count = update_all_csvs_with_missing_fields(
                        csv_missing_map,
                        update_progress
                    )

                    if failure_count > 0:
                        gui.update_status(f"⚠️ Warning: {failure_count} CSV(s) failed to update. Continuing with training...")
                        time.sleep(2)
                    else:
                        gui.update_status(f"✅ Successfully updated {success_count} CSV(s)")
                        time.sleep(1)
                else:
                    # User declined or no callback - warn and continue
                    gui.update_status(f"⚠️ Warning: {total_csvs} CSV(s) missing fields: {missing_fields_str}")
                    time.sleep(2)

            gui.update_status("1/6: Aggregating classification CSVs for Header Model...")
            all_data = []
            classified_dir = get_classified_outputs_dir(output_dir)
            scan_root = classified_dir if os.path.isdir(classified_dir) else output_dir
            for study_folder in os.listdir(scan_root):
                study_path = os.path.join(scan_root, study_folder)
                if os.path.isdir(study_path):
                    study_id = extract_study_id_from_path(study_path)
                    csv_path = os.path.join(study_path, get_study_csv_name(study_id))
                    if os.path.exists(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            all_data.append(df)
                        except Exception as e:
                            logger.warning(f"Error reading {csv_path}: {e}")

            all_data = filter_nonempty_frames(all_data)
            if not all_data:
                gui.update_status("❌ Error: No classification CSV files found to train on.")
                time.sleep(2)
                gui.close()
                return

            master_df = pd.concat(all_data, ignore_index=True)

            model_files_dir = os.path.join(output_dir, "model_files")
            os.makedirs(model_files_dir, exist_ok=True)

            master_df.to_csv(os.path.join(model_files_dir, "classification_data.csv"), index=False)

        gui.update_status("2/6: Preprocessing and detecting field types...")
        if gui.is_cancelled():
            logger.info("Header model training cancelled by user.")
            gui.close()
            return
        # Normally the negative 'none'/'other' labels are dropped so the model
        # discriminates among the target series. But if only ONE target series is
        # configured, dropping them leaves a single class and the classifier
        # returns probability 1.0 for everything (no discrimination). In that
        # case, keep the negative examples as a single 'Other' class so the model
        # can score target-vs-Other.
        cls_lower = master_df['Classification'].str.lower()
        non_neg = master_df[(cls_lower != 'none') & (cls_lower != 'other')]
        if non_neg['Classification'].nunique() >= 2:
            master_df = non_neg.copy()
        else:
            master_df = master_df.copy()
            master_df.loc[cls_lower.isin(['none', 'other']), 'Classification'] = 'Other'
            logger.info("Only one target series — training target-vs-'Other' so the "
                        "model can discriminate (kept negative examples).")
        if master_df.empty:
            gui.update_status("❌ Error: Training data is empty after removing 'None'/'other' labels.")
            time.sleep(2)
            gui.close()
            return
        if master_df['Classification'].nunique() < 2:
            gui.update_status("❌ Error: Need at least two classes to train. With a single "
                              "target series, also classify some series as 'Other'.")
            time.sleep(3)
            gui.close()
            return

        # Preprocess with automatic field type detection
        X_clean, categorical_cols, text_cols, field_types = _preprocess_df(
            master_df.drop(columns=['Classification', 'Path']),
            is_training=True
        )
        y = master_df['Classification']

        gui.update_status("3/6: Encoding categorical features and target...")
        if gui.is_cancelled():
            logger.info("Header model training cancelled by user.")
            gui.close()
            return

        model_files_dir = os.path.join(output_dir, "model_files")
        os.makedirs(model_files_dir, exist_ok=True)
        target_encoder_filename = os.path.join(model_files_dir, "label_encoder.joblib")
        if os.path.exists(target_encoder_filename):
            target_encoder = joblib.load(target_encoder_filename)
            y_encoded = target_encoder.transform(y)
        else:
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)

        feature_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded_cat = feature_encoder.fit_transform(X_clean[categorical_cols]) if categorical_cols else pd.DataFrame()
        embedded_features_dfs = encode_text_features(X_clean, text_cols)

        X_final_num = X_clean.drop(columns=categorical_cols + text_cols).copy()

        if len(categorical_cols) > 0:
            X_encoded_cat_df = pd.DataFrame(X_encoded_cat, index=X_final_num.index, columns=categorical_cols)
            X_final = pd.concat([X_final_num, X_encoded_cat_df] + embedded_features_dfs, axis=1)
        else:
            X_final = pd.concat([X_final_num] + embedded_features_dfs, axis=1)

        gui.update_status(f"4/6: Training RandomForestClassifier ({N_ESTIMATORS} trees)...")
        if gui.is_cancelled():
            logger.info("Header model training cancelled by user.")
            gui.close()
            return
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_train, y_train)

        gui.update_status("5/6: Saving model, encoders, and field types...")
        if gui.is_cancelled():
            logger.info("Header model training cancelled by user.")
            gui.close()
            return
        model_files_dir = os.path.join(output_dir, "model_files")
        os.makedirs(model_files_dir, exist_ok=True)

        model_filename = os.path.join(model_files_dir, "rf_classifier_model.joblib")
        feature_encoder_filename = os.path.join(model_files_dir, "feature_encoder.joblib")
        field_types_filename = os.path.join(model_files_dir, "field_types.joblib")

        joblib.dump(model, model_filename)
        joblib.dump(target_encoder, target_encoder_filename)
        joblib.dump(feature_encoder, feature_encoder_filename)
        joblib.dump(field_types, field_types_filename)

        gui.update_status("6/6: Evaluating model and saving results...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        final_message = f"✅ Header Model Training Complete!\nAccuracy on Test Set: {accuracy:.2f}\nModel and Encoders Saved."
        summary_path = os.path.join(model_files_dir, "rf_model_training_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(final_message)
            f.write("\n\nDetected Field Types:\n")
            for field_name, info in field_types.items():
                f.write(f"  {field_name}: {info['type']}")
                if info['type'] == 'array':
                    f.write(f" (size={info['size']})")
                f.write("\n")

        gui.update_status(final_message)
        time.sleep(5)

    except Exception as e:
        error_msg = f"An unexpected error occurred during Header Model training: {type(e).__name__}: {str(e)}"
        logger.exception(error_msg)
        gui.update_status(f"❌ {error_msg}")
        time.sleep(3)

    finally:
        gui.close()



def get_series_features(dicom_path):
    """
    Extract DICOM header features based on configured fields.
    Dynamically uses DICOM_HEADER_FIELDS from config.
    """
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
    except Exception:
        return None

    features = {}
    for display_name, dicom_attr in DICOM_HEADER_FIELDS:
        features[display_name] = safe_get_dicom_attribute(ds, dicom_attr)

    return features


def get_series_probabilities(dicom_path, model, feature_encoder, field_types=None, *args, **kwargs):
    """
    Predict probability scores for a DICOM series using the trained RF header model.

    Args:
        dicom_path: Path to DICOM file
        model: Trained RandomForest model
        feature_encoder: Trained OrdinalEncoder for categorical features
        field_types: Dictionary of field types from training (optional)

    Returns:
        np.ndarray: Probability scores for each class, or None if prediction fails
    """
    if not all([model, feature_encoder]):
        return None

    # Field types are not loaded from disk here; when None, the preprocessing
    # below falls back to automatic field-type detection.
    if field_types is None:
        try:
            field_types = None
        except Exception:
            field_types = None

    features = get_series_features(dicom_path)
    if not features:
        return None

    try:
        X_final = _build_rf_feature_matrix(pd.DataFrame([features]), feature_encoder, field_types)
        _force_single_thread_predict(model)
        return model.predict_proba(X_final)[0]
    except Exception as e:
        logger.debug(f"RF Prediction probability failed: {e}")
        return None


def _build_rf_feature_matrix(X_pred_raw, feature_encoder, field_types=None):
    """Build the model-ready feature matrix from raw header features (1+ rows).

    Shared by single- and batch-series RF inference. Text columns are embedded in
    one batched sentence-transformer call (the dominant cost), so processing all
    of a study's series at once is far faster than one call per series."""
    if field_types is not None:
        X_clean = _preprocess_df(X_pred_raw, is_training=False, field_types=field_types)
    else:
        X_clean = _preprocess_df(X_pred_raw, is_training=False)

    from .field_detection import analyze_dataframe_fields
    if field_types is not None:
        categorical_cols = [f for f, info in field_types.items() if info['type'] == 'categorical' and f in X_clean.columns]
        text_cols = [f for f, info in field_types.items() if info['type'] == 'text' and f in X_clean.columns]
    else:
        field_names = [field[0] for field in DICOM_HEADER_FIELDS]
        detected_types = analyze_dataframe_fields(X_pred_raw, field_names)
        categorical_cols = [f for f, info in detected_types.items() if info['type'] == 'categorical' and f in X_clean.columns]
        text_cols = [f for f, info in detected_types.items() if info['type'] == 'text' and f in X_clean.columns]

    X_encoded_cat = feature_encoder.transform(X_clean[categorical_cols]) if categorical_cols else []
    embedded_features_dfs = encode_text_features(X_clean, text_cols) if text_cols else []
    X_final_num = X_clean.drop(columns=categorical_cols + text_cols) if categorical_cols or text_cols else X_clean

    if len(categorical_cols) > 0:
        X_encoded_cat_df = pd.DataFrame(X_encoded_cat, index=X_final_num.index, columns=categorical_cols)
        X_final = pd.concat([X_final_num, X_encoded_cat_df] + embedded_features_dfs, axis=1)
    else:
        X_final = pd.concat([X_final_num] + embedded_features_dfs, axis=1)
    return X_final


def _force_single_thread_predict(model):
    # The model was trained with n_jobs=-1; for these small predict batches the
    # joblib ThreadPool is pure overhead (and churned/hung once per series).
    try:
        if getattr(model, "n_jobs", 1) not in (None, 1):
            model.n_jobs = 1
    except Exception:
        pass


def get_series_probabilities_batch(dicom_paths, model, feature_encoder, field_types=None):
    """Predict RF probabilities for many series at once.

    Returns a list aligned with *dicom_paths* (None where a series' header could
    not be read). Header reads are parallelized and the text embedding + RF
    predict run once over the whole batch."""
    n = len(dicom_paths)
    results = [None] * n
    if not all([model, feature_encoder]) or n == 0:
        return results

    feats = [None] * n
    workers = max(1, min(get_num_parallel_workers(), n))
    _t = time.perf_counter()   # [BENCH]
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, f in ex.map(lambda ip: (ip[0], get_series_features(ip[1])), list(enumerate(dicom_paths))):
                feats[i] = f
    except Exception:
        for i, p in enumerate(dicom_paths):
            feats[i] = get_series_features(p)
    logger.info(f"[BENCH]   RF header reads ({n} series, {workers} workers): "
                f"{(time.perf_counter()-_t)*1000:.0f} ms")  # [BENCH]

    valid = [i for i, f in enumerate(feats) if f]
    if not valid:
        return results
    try:
        _t = time.perf_counter()   # [BENCH]
        X_final = _build_rf_feature_matrix(pd.DataFrame([feats[i] for i in valid]),
                                           feature_encoder, field_types)
        logger.info(f"[BENCH]   RF feature build (preprocess + text embed + encode, "
                    f"{len(valid)} series): {(time.perf_counter()-_t)*1000:.0f} ms")  # [BENCH]
        _force_single_thread_predict(model)
        _t = time.perf_counter()   # [BENCH]
        probas = model.predict_proba(X_final)
        logger.info(f"[BENCH]   RF predict_proba ({len(valid)} series): "
                    f"{(time.perf_counter()-_t)*1000:.0f} ms")  # [BENCH]
        for j, i in enumerate(valid):
            results[i] = probas[j]
    except Exception as e:
        logger.debug(f"Batched RF prediction failed: {e}")
    return results
