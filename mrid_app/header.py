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
from .utils import safe_get_dicom_attribute, extract_study_id_from_path, get_study_csv_name
from .csv_field_updater import (
    scan_for_missing_fields,
    update_all_csvs_with_missing_fields
)


def TrainHeaderClassifierLogic(output_dir, gui, update_missing_fields_callback=None):
    """
    Train header classifier with optional missing field update.

    Args:
        output_dir: Output directory containing classification CSVs
        gui: Training GUI for status updates
        update_missing_fields_callback: Optional callback that returns True if user wants to update CSVs
    """
    if not all([pd, RandomForestClassifier, joblib, LabelEncoder, OrdinalEncoder]):
        gui.update_status("❌ Error: Missing required dependencies for Header Model.")
        time.sleep(2)
        gui.close()
        return

    try:
        # Check for missing fields in existing CSVs
        gui.update_status("Checking for missing DICOM header fields...")
        csv_missing_map = scan_for_missing_fields(output_dir)

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
        for study_folder in os.listdir(output_dir):
            study_path = os.path.join(output_dir, study_folder)
            if os.path.isdir(study_path):
                study_id = extract_study_id_from_path(study_path)
                csv_path = os.path.join(study_path, get_study_csv_name(study_id))
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        all_data.append(df)
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")

        if not all_data:
            gui.update_status("❌ Error: No classification CSV files found to train on.")
            time.sleep(2)
            gui.close()
            return

        master_df = pd.concat(all_data, ignore_index=True)
        master_df.to_csv(os.path.join(output_dir, "classification_data.csv"), index=False)

        gui.update_status("2/6: Preprocessing and detecting field types...")
        master_df = master_df[master_df['Classification'].str.lower() != 'none'].copy()
        master_df = master_df[master_df['Classification'].str.lower() != 'other'].copy()
        if master_df.empty:
            gui.update_status("❌ Error: Training data is empty after removing 'None'/'other' labels.")
            time.sleep(2)
            gui.close()
            return

        # Use new preprocessing with automatic field detection
        X_clean, categorical_cols, text_cols, field_types = _preprocess_df(
            master_df.drop(columns=['Classification', 'Path']),
            is_training=True
        )
        y = master_df['Classification']

        gui.update_status("3/6: Encoding categorical features and target...")

        target_encoder_filename = os.path.join(output_dir, "label_encoder.joblib")
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
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_train, y_train)

        gui.update_status("5/6: Saving model, encoders, and field types...")
        model_filename = os.path.join(output_dir, "rf_classifier_model.joblib")
        feature_encoder_filename = os.path.join(output_dir, "feature_encoder.joblib")
        field_types_filename = os.path.join(output_dir, "field_types.joblib")

        joblib.dump(model, model_filename)
        joblib.dump(target_encoder, target_encoder_filename)
        joblib.dump(feature_encoder, feature_encoder_filename)
        joblib.dump(field_types, field_types_filename)  # Save detected field types

        gui.update_status("6/6: Evaluating model and saving results...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        final_message = f"✅ Header Model Training Complete!\nAccuracy on Test Set: {accuracy:.2f}\nModel and Encoders Saved."
        summary_path = os.path.join(output_dir, "rf_model_training_summary.txt")
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
        error_msg = f"❌ An unexpected error occurred during Header Model training: {type(e).__name__}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        gui.update_status(error_msg)
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
        field_types: Dictionary of field types from training (optional, will try to load)

    Accepts optional extra arguments for backward compatibility (they are ignored).
    """
    if not all([model, feature_encoder]):
        return None

    # Try to load field types if not provided
    if field_types is None:
        try:
            # Try to find field_types file in same directory as model
            # Assuming model was loaded from a specific path, try to find field_types there
            # For now, we'll fall back to automatic detection if not found
            field_types = None
        except Exception:
            field_types = None

    features = get_series_features(dicom_path)
    if not features:
        return None

    X_pred_raw = pd.DataFrame([features])

    # Preprocess using field types if available
    if field_types is not None:
        X_clean = _preprocess_df(X_pred_raw, is_training=False, field_types=field_types)
    else:
        # Fallback: preprocess without field types (will auto-detect but less reliable for single sample)
        X_clean = _preprocess_df(X_pred_raw, is_training=False)

    # Determine categorical and text columns from the cleaned data
    # We need to match what was used during training
    from .field_detection import analyze_dataframe_fields

    if field_types is not None:
        # Use the field types from training
        categorical_cols = [f for f, info in field_types.items() if info['type'] == 'categorical' and f in X_clean.columns]
        text_cols = [f for f, info in field_types.items() if info['type'] == 'text' and f in X_clean.columns]
    else:
        # Try to infer (less reliable)
        field_names = [field[0] for field in DICOM_HEADER_FIELDS]
        detected_types = analyze_dataframe_fields(X_pred_raw, field_names)
        categorical_cols = [f for f, info in detected_types.items() if info['type'] == 'categorical' and f in X_clean.columns]
        text_cols = [f for f, info in detected_types.items() if info['type'] == 'text' and f in X_clean.columns]

    try:
        X_encoded_cat = feature_encoder.transform(X_clean[categorical_cols]) if categorical_cols else []
        embedded_features_dfs = encode_text_features(X_clean, text_cols) if text_cols else []
        X_final_num = X_clean.drop(columns=categorical_cols + text_cols) if categorical_cols or text_cols else X_clean

        if len(categorical_cols) > 0:
            X_encoded_cat_df = pd.DataFrame(X_encoded_cat, index=X_final_num.index, columns=categorical_cols)
            X_final = pd.concat([X_final_num, X_encoded_cat_df] + embedded_features_dfs, axis=1)
        else:
            X_final = pd.concat([X_final_num] + embedded_features_dfs, axis=1)

        y_pred_proba = model.predict_proba(X_final)[0]
        return y_pred_proba
    except Exception as e:
        print(f"RF Prediction probability failed: {e}")
        import traceback
        traceback.print_exc()
        return None
